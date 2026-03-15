from crewai import Agent, Task, Crew

class SupportCrew:

    def __init__(self, graph_executor):
        self.graph_executor = graph_executor

    def run(self, question, session_id):

        coordinator = Agent(
            role="Support Request Coordinator",
            goal="Understand the user problem and classify intent",
            backstory="Expert customer support coordinator"
        )

        retrieval_specialist = Agent(
            role="Knowledge Retrieval Specialist",
            goal="Search company knowledge base for relevant information",
            backstory="Expert in documentation and knowledge base search"
        )

        response_specialist = Agent(
            role="Customer Response Specialist",
            goal="Generate the final helpful answer for the user",
            backstory="Customer support expert"
        )

        classify_task = Task(
            description=f"Classify the intent of this support question: {question}",
            agent=coordinator
        )

        retrieve_task = Task(
            description=f"Retrieve relevant knowledge for the question: {question}",
            agent=retrieval_specialist
        )

        respond_task = Task(
            description=f"Generate a helpful response for the question: {question}",
            agent=response_specialist
        )

        crew = Crew(
            agents=[coordinator, retrieval_specialist, response_specialist],
            tasks=[classify_task, retrieve_task, respond_task]
        )

        crew.kickoff()

       
        result = self.graph_executor(question, session_id)

        return result
