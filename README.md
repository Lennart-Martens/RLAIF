This project is an attempt to automate the fine-tuning process by implementing RLAIF. The goal is to allow individuals and small businesses to create custom models efficiently and easily. These models could have strong domain knowledge, custom output formats, and stabely maintain a role.

To use the package, __init__.py in the methods folder needs to be filled out

Features:
- LLM builder
- LLM chat
- Autonomous data generation for supervised fine tuning

Roadmap:
- Data generation for a preference model
- Fully autonomous RLAIF pipeline
- Increased speed (bottleneck is API rate limit) (batch predictions?)
- Split up data generation for reliability and parallelism
- App/UI for creating custom models
- Domain knowledge data generation