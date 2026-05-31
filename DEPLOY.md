# Hannah AI — Guia de Despliegue Local (LAN)

Este documento explica como levantar el sistema Hannah para pruebas
simultaneas con multiples usuarios dentro de la misma red local.

---

## Requisitos previos

Instalar antes de continuar:

| Herramienta | Descarga | Para que sirve |
|---|---|---|
| Docker Desktop | https://www.docker.com/products/docker-desktop | Corre todos los servicios |
| Python 3.11+ | https://www.python.org/downloads | Model servers (GPU) |
| Git | https://git-scm.com | Clonar repositorios |

> Los usuarios que van a **chatear** con Hannah no necesitan instalar nada.
> Solo abren un navegador.

---

## Estructura de repositorios

El sistema usa dos repositorios que deben estar en la misma carpeta:

```
TA_PLN/
  Hannah/          <- este repositorio (backend Python + frontend)
  hannah-backend/  <- repositorio Node.js (WebSocket, voz, avatar)
```

Clonar ambos en la misma carpeta:

```
git clone <url-Hannah>        Hannah
git clone <url-hannah-backend> hannah-backend
```

---

## Primera vez: Setup

1. Abrir Docker Desktop y esperar a que este corriendo (icono en la barra de tareas).

2. Abrir CMD en la carpeta `Hannah/`:
   ```
   cd C:\...\TA_PLN\Hannah
   ```

3. Ejecutar el setup (solo una vez):
   ```
   setup.bat
   ```

   Esto construye las imagenes Docker con todas las dependencias.
   Tarda entre 5 y 15 minutos la primera vez.

4. Cuando termine, aparece el mensaje:
   ```
   Setup completo. Ahora haz doble click en arrancar.bat
   ```

---

## Uso diario: Arrancar el sistema

1. Abrir Docker Desktop.

2. Doble click en `arrancar.bat` (o desde CMD: `arrancar.bat`).

3. El script levanta todos los servicios automaticamente y muestra:
   ```
   ==========================================
     SISTEMA LISTO
   ==========================================

     URL para los usuarios:
     http://192.168.1.XX:8000

     Credenciales:
       luis   / hannah1  (admin)
       user2  / hannah2
       user3  / hannah3
       user4  / hannah4
       user5  / hannah5

     Para detener: arrancar.bat stop
   ==========================================
   ```

4. Compartir la URL con los usuarios (por WhatsApp, Slack, etc.).
   Todos deben estar conectados al mismo WiFi o red local.

---

## Acceso para usuarios

Los usuarios NO instalan nada. Solo:

1. Abrir un navegador (Chrome, Edge, Firefox).
2. Entrar a la URL que comparte el tecnico (ejemplo: `http://192.168.1.45:8000`).
3. Seleccionar su nombre en el dropdown.
4. Escribir su contrasena.
5. Click en **Entrar**.

El historial de cada usuario es independiente.
Si cierran el navegador y vuelven a entrar, se les pide login de nuevo.

---

## Servicios que corren

| Servicio | Puerto | Descripcion |
|---|---|---|
| hannah-api | 8000 | Backend Python + frontend web |
| node-gateway | 3001 | Gateway WebSocket para voz y avatar |
| redis | interno | Almacenamiento de sesiones |
| chromadb | 8010 | Base de datos vectorial (RAG) |

> Los model servers (Hannah 360M y Qwen2.5) corren **fuera de Docker**
> directamente en la maquina con GPU. Ver seccion siguiente.

---

## Model Servers (GPU) — opcional

Si los modelos ya estan entrenados y se quiere usar la IA completa,
correr en terminales separadas desde `Hannah/backend-hannah/`:

```
# Model server fast (Hannah 360M) - puerto 8001
python -m uvicorn server.hannah_model_server:app --port 8001

# Model server slow (Qwen2.5) - puerto 8003
python -m uvicorn server.qwen_model_server:app --port 8003
```

Si los model servers no estan corriendo, el sistema igualmente
funciona pero devuelve error 504 al enviar mensajes al chat.

---

## Detener el sistema

Desde CMD en la carpeta `Hannah/`:

```
arrancar.bat stop
```

O desde Docker Desktop: click en el grupo `ta_pln` y luego Stop.

---

## Solucion de problemas comunes

**"Docker no esta corriendo"**
→ Abrir Docker Desktop desde el menu inicio y esperar 30 segundos.

**"Puerto en uso"**
→ Algun servicio previo ocupa el puerto 8000 o 3001.
→ Correr `arrancar.bat stop` primero, luego `arrancar.bat`.

**La URL no carga en otro dispositivo**
→ Verificar que el dispositivo este en el mismo WiFi.
→ Revisar que el firewall de Windows no bloquee el puerto 8000.
→ Probar desactivar temporalmente el firewall de Windows Defender.

**"API tarda mas de lo normal"**
→ Ver logs: `docker compose logs hannah-api --tail 30`

---

## Credenciales por defecto

Definidas en `backend-hannah/core/auth.py`.
Cambiar los passwords antes de distribuir al equipo si se desea.

| Usuario | Password | Rol |
|---|---|---|
| luis | hannah1 | admin |
| user2 | hannah2 | user |
| user3 | hannah3 | user |
| user4 | hannah4 | user |
| user5 | hannah5 | user |

---

*Hannah AI — PLN PUCP 2026*
