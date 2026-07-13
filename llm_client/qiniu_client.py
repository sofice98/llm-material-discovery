# -*- coding: utf-8 -*-
"""Qiniu object-storage upload support for public PDF page images."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass


class QiniuUploadError(RuntimeError):
    pass


@dataclass(frozen=True)
class QiniuUploader:
    access_key: str
    secret_key: str
    bucket: str
    public_domain: str
    key_prefix: str = "pdf-pages"

    def upload_image(self, image_bytes: bytes, object_key: str) -> str:
        try:
            from qiniu import Auth, put_data
        except ImportError as exc:
            raise QiniuUploadError(
                "Missing dependency: qiniu. Run 'pip install -r requirements.txt'."
            ) from exc

        key = "/".join(
            part.strip("/") for part in (self.key_prefix, object_key) if part.strip("/")
        )
        token = Auth(self.access_key, self.secret_key).upload_token(self.bucket, key)
        response, info = put_data(token, key, image_bytes, mime_type="image/png")
        if response is None or getattr(info, "status_code", 0) != 200:
            message = getattr(info, "error", None) or str(info)
            raise QiniuUploadError(f"Qiniu upload failed for {key}: {message}")
        return f"{self.public_domain}/{key}"


def create_qiniu_uploader() -> QiniuUploader:
    values = {
        "QINIU_ACCESS_KEY": os.getenv("QINIU_ACCESS_KEY", "").strip(),
        "QINIU_SECRET_KEY": os.getenv("QINIU_SECRET_KEY", "").strip(),
        "QINIU_BUCKET": os.getenv("QINIU_BUCKET", "").strip(),
        "QINIU_PUBLIC_DOMAIN": os.getenv("QINIU_PUBLIC_DOMAIN", "").strip(),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise QiniuUploadError(
            "Missing required Qiniu environment variables: "
            + ", ".join(missing)
            + ". Please configure them in .env."
        )
    return QiniuUploader(
        access_key=values["QINIU_ACCESS_KEY"],
        secret_key=values["QINIU_SECRET_KEY"],
        bucket=values["QINIU_BUCKET"],
        public_domain=values["QINIU_PUBLIC_DOMAIN"],
        key_prefix=os.getenv("QINIU_KEY_PREFIX", "pdf-pages").strip("/"),
    )


def object_key_for_pdf_page(pdf_path: str, page_number: int) -> str:
    """Make stable keys so repeat runs overwrite the same rendered page."""
    digest = hashlib.sha256(pdf_path.encode("utf-8")).hexdigest()[:16]
    return f"{digest}/page-{page_number:04d}.png"
