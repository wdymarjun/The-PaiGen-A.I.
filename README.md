# PaiGen AI

PaiGen is a local Ollama text assistant with opt-in, on-demand camera perception.

## Camera perception

The camera feature captures a single frame only when you enter `/see`. It sends
that frame to the configured local Ollama vision model, displays a description,
and saves only the text description in the local SQLite database. It does not
record video or save image files. `/forget-seeing` removes saved descriptions
for the current session; `/clear` removes them along with chat history.

1. Install the optional camera package: `pip install opencv-python`.
2. Start Ollama and install a vision model: `ollama pull qwen2.5vl:3b`.
3. Start the companion with `python interactive_companion.py`.
4. Enter `/see` to describe the current frame, or `/see <question>` to ask about
   something visible.

Set `PAIGEN_VISION_MODEL` to another locally installed Ollama vision model if
needed. Camera index 0 is used by default. The vision model should be able to
answer from images; text-only models will not work.

PaiGen treats visual descriptions as time-stamped observations, not permanent
facts. It can refer to the most recent observations in later chat, but should
capture a fresh frame when the current scene matters.
