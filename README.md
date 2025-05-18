# AI Axur – Scraper and Inference Script

This Python script automates the process of:

1. Scraping a webpage to extract a Base64-encoded image embedded in an `<img>` tag.  
2. Sending the extracted image to a vision model API (`microsoft-florence-2-large`) with a detailed caption prompt.  
3. Submitting the model's JSON response to a validation endpoint.

---

## Overview

The script performs the following steps:

- Fetches the HTML content of a configured URL.  
- Parses the HTML to find an inline Base64 image.  
- Decodes and saves the image locally for reference.  
- Constructs the payload for the inference API using dataclasses for structured JSON.  
- Sends the payload to the inference API and receives a JSON response.  
- Submits the inference response to a separate API endpoint for validation.

---

## Code Structure

### Classes

- **ImageURL**: Represents the image URL part of the payload.  
- **Content**: Represents a message content block, which can be text or image URL.  
- **Message**: Represents a chat message containing a role and a list of content blocks.  
- **InferencePayload**: Represents the overall payload including model and messages.

### Functions

- `fetch_image_data(scraping_url: str) -> tuple[str, str]`  
  Scrapes the webpage, extracts the Base64 image data, and returns the MIME type and encoded string.

- `save_image(base64_data: str, filename: str) -> None`  
  Decodes the Base64 image data and saves it locally.

- `build_inference_payload(model: str, mime_type: str, base64_data: str) -> dict`  
  Creates the JSON payload required for the inference API, structured with dataclasses.

- `send_inference_request(payload: dict) -> dict`  
  Sends the inference request and returns the JSON response.

- `submit_response(data: dict) -> None`  
  Sends the inference JSON response to the validation API.

- `main()`  
  Coordinates the overall workflow and handles exceptions.

---

## Notes

- Sensitive data such as API tokens and URLs are stored as variables and should be managed securely outside of the source code.  
- The script uses Python's built-in dataclasses to facilitate JSON payload construction, improving readability and maintainability.  
- Proper error handling ensures that exceptions during HTTP requests or parsing are caught and logged.

---

## Author

Ronaldo Simeone Antonio – 2025  
