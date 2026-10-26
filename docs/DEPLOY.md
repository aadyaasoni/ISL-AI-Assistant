# Deploy to Hugging Face Spaces

## Create the Space

1. Sign in to [Hugging Face](https://huggingface.co/).
2. Select **New Space**.
3. Choose a Space name and visibility.
4. Select **Docker** as the Space SDK.
5. Create the Space and copy its Git URL from the Space page.

## Push this repository

From the repository checkout, add the Space remote and push the deployment branch (or `main` after it contains this setup):

```bash
git remote add hf https://huggingface.co/spaces/<your-username>/<your-space-name>
git push hf deploy-today:main
```

If Hugging Face requests authentication, use the Hugging Face CLI login flow or a credential manager. Do not put an access token in the remote URL or commit it.

The Dockerfile listens on `0.0.0.0` and defaults to port `7860`. In the Space **Settings**, set `app_port` to `7860` if it is not already configured. The runtime bundle (model weights, label map, MediaPipe task files, metadata, and extracted sample landmarks) is committed in this repository and copied into the image; the app does not download model or dataset files at startup. The default failure log is `/tmp/failure_events.jsonl`; set `ISL_FAILURE_LOG_PATH` to an alternate writable path if needed.

The Space may take longer to build the first time because CPU PyTorch and MediaPipe are installed during the image build. After the build, `/api/health` is the container health check.

## Camera access

Use the direct `https://<your-space-name>.hf.space` URL, not the embedded iframe on `huggingface.co/spaces/...`. Browser webcam access requires HTTPS and user permission. Camera frames are processed by the service into landmarks and are not stored.

## Secondary option: Render

The root `render.yaml` is a minimal Docker web-service blueprint. Import this repository in Render and allow it to build the Dockerfile. The app listens on the platform-provided `PORT`.

## Space README front matter

Copy [`deploy/hf_space_README.md`](../deploy/hf_space_README.md) to the Space repository as its `README.md` (do not replace this repository's GitHub README). It provides the Hugging Face YAML metadata and an honest feature description.
