# LLM-World
* Generative AI
   * Generate Chat
   * Fine-Tuning
* One-Shot AI Agent
  * LangChain Agent
* Multi-Agent (Agentic Workflow)
  * LangGraph Agent
* Open Weight
  * https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html
  * Fine-Tuning


# Dockerfile
* docker build -t llm-image . (build iamge)
* docker run -it --name llm-container --gpus all -v host-path:container-path llm-image:latests (build/run container) [-p 8888:8888] if running server

