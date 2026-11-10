# Inference Router

One OpenAI-compatible endpoint for a company. It routes each request to any OpenAI-compatible
upstream (NVIDIA, OpenRouter, Anthropic, OpenAI, a rented Vast.ai GPU, or the company's own
machines through a tunnel), gives every person or app their own API key, and can mask personal
data or keep personal requests on private hardware. It never stores message content.

Status: design stage. Read [docs/DESIGN.md](docs/DESIGN.md).

The router core it starts from currently lives in
[`openshell_controller/scripts/pii-router`](https://github.com/ivobrett/openshell_controller/tree/gatewaydashboard/scripts/pii-router)
and moves here in phase 0.
