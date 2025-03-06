# Building an LLM ChatBot

## How to Run It

### Running Locally
The default method is to deploy the LLM locally and interact with it directly within the same script:
```bash
python main_local.py
```

### Running as a Web Service
Alternatively, you can deploy the model as a web service using Flask, allowing it to handle requests on a specified port.

First, launch the server:
```bash
python -m llm.flask.server
```
This will start the server on `localhost` at port **8000**.

Then, you can send queries to the LLM using **Postman** or any other tool that supports **POST** requests. For convenience, we provide a script that automates this process. It sends a request and stores the response in the `responses` folder:
```bash
python main_client_flask.py
```

### Collaborators
- [Unai Ruiz](https://github.com/Ruicky8)
