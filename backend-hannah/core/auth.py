"""
core/auth.py
------------
Autenticación estática para pruebas locales con 5 usuarios.
NO usar en producción ni exponer a internet.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from jose import jwt, JWTError
from fastapi import Depends, HTTPException, Header

from config import settings

# ── 5 usuarios estáticos ──────────────────────────────────────────────
# Cambiar passwords antes de distribuir al equipo.
STATIC_USERS: dict[str, dict] = {
    "luis":  {"password": "hannah1", "tenant_id": "local", "role": "admin"},
    "user2": {"password": "hannah2", "tenant_id": "local", "role": "user"},
    "user3": {"password": "hannah3", "tenant_id": "local", "role": "user"},
    "user4": {"password": "hannah4", "tenant_id": "local", "role": "user"},
    "user5": {"password": "hannah5", "tenant_id": "local", "role": "user"},
}

ALGORITHM = "HS256"


@dataclass
class TokenData:
    user_id: str
    tenant_id: str
    role: str = "user"


def create_token(username: str, password: str) -> str:
    """Valida credenciales y genera un JWT sin expiración."""
    user = STATIC_USERS.get(username)
    if not user or user["password"] != password:
        raise ValueError("Credenciales incorrectas")
    payload = {
        "sub":       username,
        "tenant_id": user["tenant_id"],
        "role":      user["role"],
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def verify_token(authorization: str = Header(...)) -> TokenData:
    """FastAPI Dependency. Extrae tenant_id del JWT Bearer token."""
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError("Scheme inválido")
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
        return TokenData(
            user_id=payload["sub"],
            tenant_id=payload["tenant_id"],
            role=payload.get("role", "user"),
        )
    except (JWTError, ValueError, KeyError, AttributeError):
        raise HTTPException(status_code=401, detail="Token inválido o expirado")
