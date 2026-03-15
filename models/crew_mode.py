from typing import Any


class CrewCoordinator:
    def __init__(self, provider):
        self.provider = provider
        self.available = False
        try:
            from crewai import Agent, Crew, Task
            self.Agent = Agent
            self.Crew = Crew
            self.Task = Task
            self.available = True
        except Exception:
            self.Agent = None
            self.Crew = None
            self.Task = None

    def run(self, question: str, retrieval: list[dict[str, Any]], intent: str, sentiment: str):
        if not self.available:
            return None

        llm = self.provider.llm if hasattr(self.provider, "llm") else None
        if llm is None:
            return None

        context = "\n".join([f"{item['question']}: {item['answer']}" for item in retrieval])

        researcher = self.Agent(
            role="Policy specialist",
            goal="Find the best support policy answer",
            backstory="Customer support policy specialist",
            allow_delegation=False,
            verbose=False,
            llm=llm
        )

        responder = self.Agent(
            role="Response specialist",
            goal="Write a short customer-safe answer",
            backstory="Customer support response specialist",
            allow_delegation=False,
            verbose=False,
            llm=llm
        )

        task_one = self.Task(
            description=f"Analyze the user issue. Intent: {intent}. Sentiment: {sentiment}. Context: {context}. Question: {question}",
            expected_output="A concise support finding.",
            agent=researcher
        )

        task_two = self.Task(
            description="Write the final answer using the finding from the previous task.",
            expected_output="A concise final answer.",
            agent=responder
        )

        crew = self.Crew(
            agents=[researcher, responder],
            tasks=[task_one, task_two],
            verbose=False
        )

        result = crew.kickoff()
        return str(result)