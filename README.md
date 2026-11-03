# FLUX.3 Klein on Modal

[FLUX.3 Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) (Distilled or [Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base)) as an authenticated HTTP endpoint on Modal. One container runs the whole pipeline on one B200: prompt upsampler (vLLM), text encoder, FP8R transformer and autoencoder (SGLang). Text to image, editing with reference images, layout boxes, captions, JPEG or PNG. Scales to zero and restores from a GPU memory snapshot.

| | Distilled | Base |
|---|---|---|
| Defaults | 16 steps, 1024x1024 | 50 steps, guidance 3.5, 1024x1024 |
| Per picture, warm | 1.7 s | 4.0 s |
| Pictures per minute on one B200 | 35 (p95 3.0 s) | 11 (p95 4.4 s) |
| Cold start with a snapshot, to the first picture | 33 s | 33 s |
| Cold start without one (first start after a deploy) | 9 min | 8 min |
| One B200 kept warm | $6.89 per hour | same |

`app.py` is the deployment (configuration, image build, container start, Modal server), `flux3_api.py` the request handling, `requirements-pup.txt` the upsampler's locked environment.

For more than about 0.5 pictures per second, or several models, [flux-3-klein-shared-upsampler](https://github.com/nicolaslara/flux-3-klein-shared-upsampler) runs the upsampler once for all workers: cold starts of 17 s instead of 33 s and a third less per picture beyond one worker.

## Quick start

Accept the terms on Hugging Face for the model and the [prompt upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup), then:

```
uv sync
uv run modal setup
uv run modal secret create huggingface HF_TOKEN=hf_...
uv run modal deploy app.py
```

The first deploy builds the image with the weights inside (34 GB). Deploy prints the URL; export it with a proxy auth token from the Modal dashboard (Settings, Proxy Auth Tokens):

```
export URL=https://<workspace>-<environment>--flux3-klein-distilled-server.us-east.modal.direct
export MODAL_KEY=wk-...
export MODAL_SECRET=ws-...
```

While no container is up, every request gets an immediate 503 and starts one. A container whose engine call failed also answers 503 and then exits, and Modal replaces it. So poll `/wake` until it answers 200, then send `/generate` once; a non-200 means no picture was made.

```
until curl -sf -o /dev/null -X POST -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" $URL/wake; do sleep 5; done
curl -sf -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" \
  --json '{"prompt": "Two blue mugs on a wooden table by a window", "seed": 42}' \
  $URL/generate -o image.jpg
```

## API

`POST /generate`, JSON body. Unknown fields are rejected.

| field | default | notes |
|---|---|---|
| `prompt` | | Expanded by the upsampler. Exactly one of `prompt` and `caption` |
| `caption` | | An already expanded caption, used as is |
| `images` | `[]` | Up to ten base64-encoded reference images, each up to 10 MB |
| `width`, `height` | 1024, or the first reference's size | Multiples of 16, up to 4096 per side and 4 megapixels |
| `seed` | `0` | Same seed and same `caption`, same picture on the same GPU model; a `prompt` is expanded afresh each time, so send back the `X-Caption` header as `caption` to reproduce a picture |
| `steps` | 16 Distilled, 50 Base | 1 to 100 |
| `guidance` | 3.5 on Base | 0 to 20. Not used by Distilled |
| `output_format` | `jpeg` | `jpeg` (quality 95) or `png` |

The response is the picture; `X-Pup-Ms` and `X-Image-Ms` headers give the expansion and generation time. 422 is an invalid request (the body says what), 429 a container that briefly exceeded its four in-flight requests (retry), 502 an engine failure, 503 no container up or a container being replaced, 504 the 300 s deadline.

Editing: the reference pictures go base64-encoded in `images`. The body is too large for a command-line argument, so write it to a file:

```
printf '{"prompt": "Make the mugs red", "images": ["%s"], "output_format": "png"}' "$(base64 < image.jpg | tr -d '\n')" > edit.json
curl -sf -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" --json @edit.json $URL/generate -o edit.png
```

Layouts: end the prompt with a JSON list of boxes, `[y_min, x_min, y_max, x_max]` on a 0 to 1000 canvas, origin top left:

```
{"prompt": "A blue mug on the left [{\"id\": \"mug\", \"bbox\": [200, 100, 800, 450], \"desc\": \"blue mug\"}]", "seed": 42}
```

## Cold starts and scaling

A container stops 120 s after its last request, and the next request starts one: about half a minute with a snapshot, about nine minutes without one (the first start after a deploy, or on a new kind of worker). To keep a container up:

```
uv run modal run app.py --warm 1
```

A container renders one picture at a time, lets the next request's encoding overlap it, and holds up to four requests. Beyond that Modal queues and adds containers, up to `REPLICAS`.

## Configuration

Environment variables read at deploy time, for example `MODEL=base WARM=1 uv run modal deploy app.py`:

| variable | default | notes |
|---|---|---|
| `MODEL` | `distilled` | `distilled` or `base`; separate apps |
| `GPU` | `B200` | `B200`, `H200`, `H100` or `H100!`; see the GPU table below. H100 limits requests to 1536x1536 and two references |
| `WARM` | `0` | Containers kept up; `modal run app.py --warm N` changes it without a redeploy |
| `REPLICAS` | `1` | Most containers, up to 16 |
| `IDLE_SECONDS` | `120` | How long an idle container stays up |
| `SNAPSHOT` | `1` | `0` makes every start fresh |
| `HIGH_RES` | `0` | Distilled on B200 only: text to image up to 6256 per side, one request at a time |

A redeploy with the same code and settings keeps the snapshot; changing either makes the next start a fresh one.

## Measurements

One B200, 1024x1024, prompt expansion included. "Warm" is one request at a time on a running container. "Under load" is a steady stream of independent requests (128 for Distilled, 64 for Base, none failed), so the percentiles include waiting for the container.

| model | one picture, warm | pictures per minute under load | p50 / p95 / p99 under load |
|---|---|---|---|
| Distilled | 1.74 s | 35 | 2.02 / 3.04 / 3.18 s |
| Base | 4.02 s | 11 | 4.07 / 4.38 / 4.40 s |

How long one picture takes depends on the request. Distilled, warm, one request at a time:

| request | one picture |
|---|---|
| Text to image, 1024x1024 | 1.7 s |
| Edit with one reference image, 1024x1024 output | 2.5 s |
| Edit with two reference images | 3.5 s |
| Edit with ten reference images | 16 s |
| Text to image, 2048x2048 | 4.7 s |
| Text to image, 4096x4096 (`HIGH_RES=1`) | 35 s |

Cold starts, medians:

| | Distilled | Base |
|---|---|---|
| With a snapshot, until the first picture | 33 s | 33 s |
| Without a snapshot, until the container is ready | 558 s | 460 s |

Other GPUs. These numbers cover the image engine only (captions supplied, so no prompt expansion), 1024x1024, two requests in flight; they compare GPUs rather than describe the full endpoint:

| GPU | Distilled pictures/min | Base pictures/min | kept warm, per hour |
|---|---|---|---|
| H100 | 58 | 10 | $4.59 |
| H200 | 61 | 11 | $5.18 |
| B200 | 120 | 23 | $6.89 |

B200 makes twice the pictures per hour for 1.3 to 1.5 times the price, so it is both the fastest and the cheapest per picture. H100 is for when the hourly cost of a warm container matters more than latency (3.2 s per Distilled picture instead of 1.7 s).
