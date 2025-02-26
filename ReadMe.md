# Unsupervised GNN Anomaly Detection in Manufacturing

This repository contains an end-to-end solution for detecting anomalies in a manufacturing process using Graph Neural Networks (GNN). The project leverages Graph Attention Networks (GAT) for improved performance and interpretability, and includes:

- A **FastAPI** backend that loads a trained model and provides a prediction API.
- A **Streamlit** frontend that allows users to input machine details and view anomaly detection results.
- **Dockerization** of the FastAPI backend for easy deployment.
- A saved trained model (`gat_model.pth`) for inference.

## Table of Contents

- [Project Overview](#project-overview)
- [Features and Enhancements](#features-and-enhancements)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
  - [Running the FastAPI Backend with Docker](#running-the-fastapi-backend-with-docker)
  - [Running the Streamlit Frontend](#running-the-streamlit-frontend)
- [Deployment Options](#deployment-options)
- [Acknowledgments](#acknowledgments)

## Project Overview

This project focuses on detecting anomalies in the manufacturing process. The manufacturing process follows a sequential flow:
- **Task 1:** Machine 1 operation.
- **Task 2:** Machine 2 operation (starting after Task 1).
- **Task 3:** Machine 3 operation (starting after Task 2).

The system leverages:
- **Tabular data conversion into a graph representation:** Nodes represent machine states (including status and worker count), and edges capture sequential and temporal dependencies.
- **Graph Attention Networks (GAT):** Enhances the model by dynamically weighing the importance of neighboring nodes.
- **Anomaly Scoring:** Reconstruction errors are computed per node to identify deviations.
- **Visualization:** A Streamlit app displays the results and reconstruction error trends over time.

## Features and Enhancements

- **Advanced GNN Architecture:** Utilizes GAT layers for dynamic neighbor attention.
- **Machine-Worker Interaction Nodes:** Incorporates both machine and worker nodes for richer representation.
- **Per-node Anomaly Scoring:** Computes reconstruction errors per node to pinpoint anomalies.
- **User-friendly Dashboard:** Streamlit app for real-time anomaly detection and visualization.
- **Dockerized FastAPI Backend:** Simplifies deployment and scaling.
- **Model Persistence:** Trained model is saved (`gat_model.pth`) and loaded for predictions.

## Project Structure

```bash
├── main.py               # FastAPI backend
├── streamlit_app.py      # Streamlit frontend
├── gat_model.pth         # Saved trained model
├── training_script.py    # (Optional) Training script
├── Dockerfile
├── README.md
└── requirements.txt
```
 

## Installation and Setup

### Prerequisites

- **Docker:** To run the FastAPI backend in a container.
- **Python 3.9+** and **pip:** For running the Streamlit app locally.
- **Virtual Environment:** Recommended for Python dependencies.

### Installing Dependencies

1. **Clone the Repository:**

   ```bash
   git clone <your-repository-url>
   cd <your-repository-directory>


2. **Commands to run the application:**

    ```bash
    docker build -t fastapi-anomaly 
    docker run -p 8000:8000 fastapi-anomaly
    streamlit run streamlit_app.py
