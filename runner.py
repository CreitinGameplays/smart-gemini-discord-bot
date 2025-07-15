import asyncio
import sys
import json
import os
import traceback

# Simple file-based communication
INPUT_FILE = "/app/input.json"
OUTPUT_FILE = "/app/output.json"

async def execute_code(code):
    # Redirect standard output and error to capture results
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = sys.stderr = output = StringIO()
    try:
        exec(code, {}, {})
    except Exception:
        traceback.print_exc(file=sys.stderr)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        return output.getvalue()

async def main():
    while True:
        if os.path.exists(INPUT_FILE):
            with open(INPUT_FILE, "r") as f:
                try:
                    data = json.load(f)
                    code = data.get("code")
                except json.JSONDecodeError:
                    print("Invalid JSON in input file.")
                    continue

            if code:
                result = await execute_code(code)
                with open(OUTPUT_FILE, "w") as f:
                    json.dump({"result": result}, f)

            os.remove(INPUT_FILE)
        await asyncio.sleep(0.1)  # Check for new input every 100ms

if __name__ == "__main__":
    from io import StringIO
    asyncio.run(main())