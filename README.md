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
## Build Image
* docker build -t llm-image . (build image)
## Build Container
* docker run -it --name llm-container --gpus all -v host-path:container-path llm-image:latests [-p 8888:8888] (if running server)
  * docker run --name llm-containr --gpus all -v /home/marfok/CLONED_REPOS/LLM-World/Notebooks/:/home/notebooks -v /home/marfok/CLONED_REPOS/LLM-World/Files/:/home/files -p 8888:8890 llm-image:latest   
### Shell Container
* docker exec -it (container-id) bash  
  * start server: jupyter notebook --ip 0.0.0.0 --port 8090 --allow-root --no-browser     
