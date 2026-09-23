"""Talk to the bot in the terminal.

Run:  python -m scripts.chat
"""

from __future__ import annotations

from src.chatbot import Conversation, Intent, RecipeChatbot
from src.chatbot.responses import GREETING


def main() -> None:
    bot = RecipeChatbot.from_index()
    conversation = Conversation()

    status = ("LLM: on (%s)" % bot.settings.llm_model if bot.llm_enabled
              else "LLM: off -- template replies")
    print(f"\n{GREETING}\n({status}. Ctrl-C or \"bye\" to quit.)\n")
    while True:
        try:
            message = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbot > Happy cooking!")
            return
        if not message:
            continue

        reply = bot.respond(message, conversation)
        print(f"\nbot > {reply.text}\n")
        if reply.intent is Intent.GOODBYE:
            return


if __name__ == "__main__":
    main()
