# Enterprise Chatbot System Design - Summary

This system design project is a highly available and fault tolerant enterprise chatbot system that seamlessly integrates multiple communication channels and maintains robust security and scalability. The deeply interconnected architecture incorporates a modular design with distinct layers for communication, intelligence, data management, and model maintenance. It is a self-improving system; the data annotation and review services allow the chatbot to constantly learn and adapt based on real-world usage, closing the feedback loop and enhancing performance over time. This design delivers a secure, reliable chatbot solution that meets the demanding uptime and growth requirements of modern business operations.

# Enterprise Chatbot - System Overview

The System Overview diagram contains the basic components of the system. Here I will explain the functions and motivations of each part of the system and the interfaces between them.

* External Communication Services - Twilio, Whatsapp, Javascript SDK on a website, etc.
* Conversation Orchestrator Service - web application with admin site, connections to external communication services, uses Raw Chat Database as datastore, communicates with Chat Backend asynchronously through task queues
* Chat Backend Service - stateless application (conversation state is always passed in), task queue, no database connection, does dialog management and response generation
* ML Model Inference Service - web endpoint that accepts user input and returns model response, requires access to saved models
* Raw Chat Database - relational DB with tables for messages and conversations at minimum, may include other data as required by the business, such as user/organization information or other integration data (e.g. hooking into an Applicant Tracking System for a jobs/recruiting chatbot, or into an Electronic Medical Records system for a healthcare chatbot)
* Chat Data ETL - de-identification, anonymization, aggregation (e.g. from different orgs), etc.
* Internal Chat Database - contains data ETL'd from Raw Chat Database, also includes info on ML model dataset versioning and data annotations
* Data Review and Model Maintenance Service - web application with admin site, source of truth and versioning for ML model datasets, uses Internal Chat Database as datastore, sends data to ML Training Service for training models
* Data Annotation Service - web application for data annotation and labeling, e.g. Prodigy, uses Internal Chat Database as datastore
* ML Model Training Service - web endpoint that accepts datasets, trains model, and updates the model store used by the ML Model Inference Service

# LLM-powered Enterprise Chatbot

Many of the components of a traditional Conversational AI system would be the same in a similar Generative AI-based system built on top of LLMs. Here is a summary of some major differences:

* Text Generation Service - replaces or is deployed alongside the ML Model Inference service, handles prompt creation, retrieval for RAG services, and calls to LLM for text generation (using an API or LLM you host and manage), requires access to datastores such as vector/graph databases for embeddings and knowledge graph retrieval
* Raw Chat Database - possible new fields for AI-generated text (LLM w/ version, number of tokens in prompt/output, etc.)
* Chat Data ETL and Chat Database - handles any new fields added to Raw Chat Database
* Data Review and Model Maintenance Service - reorganized to handle reviewing AI-generated text and LLM prompts, focus will be less on intent classification and named entity recognition accuracy and more on quality of AI-generated text, retrieval, and prompting, requires access to the same datastores as the Text Generation Service, may want to add LLM text generation and retrieval testing directly to this application. While I would advise against taking the humans completely out of the loop, using techniques like LLM-as-a-judge can help accelerate conversation reviews and identify/prioritize areas for improvement.
* Data Annotation Service - LLMs can be put to work annotating data if needed

# Enterprise Chatbot - AWS Implementation

A system like this may be deployed on-prem, in the cloud, or using a combination of the two. Each component has many choices of implementation, and even once the technology stack is chosen, there are still more decisions (e.g. cloud-managed vs. self-managed services like databases or container orchestration). Here I will present one option for an AWS Cloud deployment. This design satisfies many of the requirements an enterprise chatbot might have, such as high availability and responsiveness, fault tolerance, and scalability.

## Questions? Comments?

If you'd like to discuss anything related to this project, you can reach me through email or LinkedIn.

* [dev@jasonbrazeal.com](mailto:dev@jasonbrazeal.com)
* [https://www.linkedin.com/in/jasonbrazeal](https://www.linkedin.com/in/jasonbrazeal)
* [https://jasonbrazeal.com](https://jasonbrazeal.com)
