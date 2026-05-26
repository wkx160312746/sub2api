# image2 TOS URL Output Design

## Context

The image2 synchronous API path and the asynchronous image task path both ultimately use `OpenAIGatewayService.ForwardImages`. The async worker records the same HTTP response body produced by that forwarding path, so response rewriting inside the image forwarding layer can cover both sync and async behavior without changing the task storage model.

The BYTS TOS package supplied in `byts-tos-main.zip` exposes a signed business API for upload and read links:

- `POST /api/tos/upload-url` returns a presigned TOS `PUT` URL.
- The gateway uploads decoded image bytes to that URL.
- `POST /api/tos/read-link` returns the URL that should be sent back to clients.

The target bucket is `open-api`.

## Decision

Use the BYTS TOS signed business API, not the admin token API. The gateway will upload image2 outputs before sending the final response to clients. When TOS is configured, image responses that currently contain `b64_json` or `data:image/...;base64,...` will be rewritten to URL responses backed by TOS. When TOS is not configured, existing behavior remains unchanged.

## Configuration

Add a small TOS image storage config section under gateway configuration, with environment overrides following the existing config loader conventions:

- `gateway.image_tos.enabled`
- `gateway.image_tos.base_url`
- `gateway.image_tos.client_id`
- `gateway.image_tos.sm2_private_key`
- `gateway.image_tos.bucket`, default `open-api`
- `gateway.image_tos.prefix`, optional object prefix
- `gateway.image_tos.upload_url_expires_seconds`, default `900`
- `gateway.image_tos.read_link_expires_seconds`, optional; when absent, request a public read link

The feature is active only when enabled and the required connection fields are present.

## Architecture

Introduce a focused service helper, `TOSImageStorage`, owned by the OpenAI image forwarding layer. Its responsibilities are:

- Sign BYTS TOS JSON requests with SM2/SM3 according to the zip documentation.
- Ask TOS for a presigned upload URL.
- Upload decoded image bytes with `PUT`.
- Ask TOS for the read link.
- Return the final URL plus bucket/key metadata for logging or tests.

The helper should not know about OpenAI response formats. OpenAI response rewriting stays near the existing image response handlers, where the code already understands `b64_json`, data URLs, streaming completion events, and OAuth Responses-derived payloads.

## Data Flow

For non-streaming image responses:

1. Read the upstream response body as today.
2. Detect image payloads in `data[].b64_json` and `data[].url` values that are data URLs.
3. Upload each image to TOS.
4. Replace each item with `url: <tosReadUrl>` and remove `b64_json`.
5. Write the rewritten JSON response.

For OAuth Responses image conversion:

1. Convert upstream Responses output into OpenAI Images API response data as today.
2. Before writing the response, run the same JSON rewrite helper.
3. Async image tasks automatically persist the rewritten response because they use the same recorder output.

For streaming image responses:

1. Leave partial image events unchanged so clients can still show progress.
2. For final completed image events, upload the final image payload before emitting the completed event.
3. Emit completed payloads with `url` instead of base64 image data.

## Error Handling

If TOS is disabled or incomplete, do not change image output behavior.

If TOS is enabled and an upload or read-link step fails, fail the image request rather than returning base64. This keeps the configured contract clear: once enabled, successful image2 responses contain usable TOS URLs.

Errors should be sanitized in client responses and include enough server-side log context to identify the TOS step, bucket, and generated object key when available.

## Testing

Add unit tests around the response rewrite helper and TOS client using `httptest`:

- Signed requests include the required BYTS headers and use bucket `open-api`.
- A `b64_json` image is uploaded and rewritten to `url`.
- A data URL in `url` is uploaded and rewritten to a normal URL.
- Existing URL responses are left unchanged.
- TOS disabled leaves responses unchanged.
- TOS enabled with upload failure returns an error.
- Async task result handling is covered by a focused test or by verifying the shared forwarding response helper is used before recorder output is captured.

Run targeted Go tests for the OpenAI image service package, then broader handler/service tests if the edits touch shared config or forwarding behavior.
