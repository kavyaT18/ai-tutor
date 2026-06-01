
from groq import Groq
import os
import yaml

class StudyAgents:

    def __init__(
        self,
        topic,
        subject_category,
        knowledge_level,
        learning_goal,
        time_available,
        learning_style,
        model_name="llama-3.3-70b-versatile"
    ):

        self.topic = topic
        self.subject_category = subject_category
        self.knowledge_level = knowledge_level
        self.learning_goal = learning_goal
        self.time_available = time_available
        self.learning_style = learning_style
        self.model_name = model_name

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

        with open("prompts.yaml", "r") as f:
            config = yaml.safe_load(f) or {}

        self.personas = config.get("personas", {})
        self.learning_styles = config.get("learning_styles", {})

    def _build_context(self):

        style_info = self.learning_styles.get(
            self.learning_style,
            {}
        )

        return f"""
TOPIC: {self.topic}

STUDENT PROFILE:
- Knowledge level: {self.knowledge_level}
- Goal: {self.learning_goal}
- Time available: {self.time_available}
- Learning style: {self.learning_style}

LEARNING STYLE NOTES:
{style_info.get("description", "")}
"""

    def generate(self, persona_name, user_prompt, chat_history=None):

        persona = self.personas.get(
            persona_name,
            {}
        )

        system_prompt = persona.get(
            "system_prompt",
            ""
        )

        full_system = f"""
{system_prompt}

{self._build_context()}
"""

        messages = [
            {
                "role": "system",
                "content": full_system
            }
        ]

        if chat_history:
            messages.extend(chat_history)

        messages.append({
            "role": "user",
            "content": user_prompt
        })

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7
        )

        return response.choices[0].message.content
