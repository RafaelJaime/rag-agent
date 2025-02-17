import resend
from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel
import os

class EmailInput(BaseModel):
    to: str = Field(..., description="Email address of recipient")
    subject: str = Field(..., description="Subject of the email")
    conversation_summary: str = Field(..., description="Summary of the conversation to include in the email")

class EmailTool(BaseTool):
    name: str = "EmailSummaryTool"
    description: str = "Send an email to a specified recipient with a given subject and a summary of the conversation."
    args_schema: type[BaseModel] = EmailInput
    return_direct: bool = False  # if True the agent will stop and return the result directly to the user

    def _run(self, to: str, subject: str, conversation_summary: str) -> str:
        resend.api_key = os.getenv('RESEND_TOKEN')

        r = resend.Emails.send({
            "from": "onboarding@resend.dev",
            "to": to,
            "subject": subject,
            "html": conversation_summary
        })

        return f"Email with conversation summary sent successfully to {to} with subject '{subject}'"
