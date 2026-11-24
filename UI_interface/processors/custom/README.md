# Custom processors

The generic processors handle every task mode. Write a custom processor only
when some documents need logic the generic flow cannot express: special
preprocessing, several endpoints, a different prompt flow, or unusual parsing.

## Contract

```python
from processors.base_processor import BaseProcessor, ProcessingContext, ProcessOutput, render_prompt

class MyProcessor(BaseProcessor):
    name = "my_processor"

    def process(self, image, ctx: ProcessingContext) -> ProcessOutput:
        prompt = render_prompt(ctx.prompt, ctx.template, ctx.task.fields)
        resp = self.call_model(image, prompt, ctx)          # uses the selected model's adapter
        result = ...                                        # your parsing
        status, flags = self.finalize(result, ctx)          # runs the task's review rules
        return ProcessOutput(status=status, result=result, raw_text=resp.raw_text, flags=flags,
                             latency_ms=resp.latency_ms)
```

`ctx.task` holds the mode, the user's fields, template, schema and rules.
Status values: `completed`, `needs_review`, `parse_failed`, `failed`.
Always keep the model's original text in `raw_text`; never return `{}` as success.

## Registering

Add an entry to `config/processor_registry.py`:

```python
CUSTOM_PROCESSORS = {
    "Cheque words-vs-figures check": {
        "path": "processors.custom.my_processor:MyProcessor",
        "modes": ["schema_based"],
    },
}
```

It then appears in the sidebar under *Processing pipeline* for those modes.
A processor that belongs to one model (like the Qwen-script processor) is
set as `processor=` on that model's entry in `config/model_registry.py`.
`qwen_scripts.py` is a complete example.
