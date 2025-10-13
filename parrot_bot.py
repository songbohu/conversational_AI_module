from dialogue_system import DialogueSystem

class ParrotBot(DialogueSystem):

    def chat(self, utterance: str) -> dict:
        return {
            "text": utterance,
            "action": {"type": "echo"}
        }

if __name__ == "__main__":
    bot = ParrotBot()
    bot.start_a_chat()
