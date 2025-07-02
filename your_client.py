import os
import json
import subprocess
import threading

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

class TerraformMCP:
    def __init__(self, server_cmd=None):
        cmd = server_cmd or ["terraform-mcp-server", "stdio"]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._id = 0
        self._lock = threading.Lock()

    def _rpc(self, method, params):
        with self._lock:
            self._id += 1
            # send JSON-RPC
            req = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()

            # debug: concise request line
            print(f">>> RPC {method} (id={self._id}) params={{{', '.join(params.keys())}}}")

            # read raw response
            raw = self.proc.stdout.readline().strip()
            # parse & extract result or error
            parsed = json.loads(raw)
            result = parsed.get("result", parsed.get("error", {}))
            # debug: show which top‐level keys came back
            print(f"<<< RPC {method} (id={self._id}) response_keys={list(result.keys())}\n")

            return parsed

    def initialize(self):
        return self._rpc("initialize", {})

    def list_tools(self):
        out = self._rpc("tools/list", {})
        return out.get("result", {}).get("tools", [])

    def call_tool(self, name, arguments):
        out = self._rpc("tools/call", {"name": name, "arguments": arguments})
        return out.get("result")


class ChatWithMCP:
    def __init__(self, mcp: TerraformMCP):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.mcp = mcp
        self.functions = [
            {
                "name": "list_tools",
                "description": "List available Terraform MCP tools",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "call_tool",
                "description": "Invoke a Terraform MCP tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                },
            },
        ]

    def run(self):
        history = []
        print("Type ‘exit’ to quit.")
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() in ("exit", "quit"):
                break

            history.append({"role": "user", "content": user_input})
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=history,
                functions=self.functions,
                function_call="auto",
            )
            msg = resp.choices[0].message

            if msg.function_call:
                fname = msg.function_call.name
                args = json.loads(msg.function_call.arguments)

                if fname == "list_tools":
                    result = self.mcp.list_tools()
                else:
                    result = self.mcp.call_tool(args["name"], args["arguments"])

                # append the function‐call
                history.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": fname,
                        "arguments": msg.function_call.arguments
                    }
                })
                # append function result
                history.append({
                    "role": "function",
                    "name": fname,
                    "content": json.dumps(result)
                })

                # final assistant reply
                follow = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=history,
                )
                reply = follow.choices[0].message.content.strip()
                print("MCP ➔", reply)
                history.append({"role": "assistant", "content": reply})
            else:
                reply = msg.content.strip()
                print("LLM ➔", reply)
                history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    raw = os.getenv("MCP_CMD")
    server_cmd = raw.split() if raw else None

    mcp = TerraformMCP(server_cmd)
    print("Initializing MCP server…", mcp.initialize())
    print("Available tools:", [t["name"] for t in mcp.list_tools()])
    ChatWithMCP(mcp).run()

