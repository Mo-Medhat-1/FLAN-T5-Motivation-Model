# FLAN-T5 Motivation Model

## Project Overview

This project presents an innovative Artificial Intelligence model designed to enhance user focus and productivity by generating personalized motivational messages, or "nudges," tailored to their emotional state. Built upon the powerful **FLAN-T5** architecture and fine-tuned using **LoRA (Low-Rank Adaptation)** and **PEFT (Parameter-Efficient Fine-Tuning)** techniques, this model aims to provide timely encouragement to users, helping them stay engaged with their tasks for extended periods.

## Key Features

*   **Personalized Nudge Generation:** The model generates encouraging messages dynamically, adapting to whether the user is experiencing low motivation or high enthusiasm.
*   **FLAN-T5 Base Model:** Leverages the robust capabilities of Google's FLAN-T5 for sequence-to-sequence tasks, ensuring high-quality text generation.
*   **LoRA & PEFT Fine-tuning:** Utilizes parameter-efficient fine-tuning methods to adapt the large language model to the specific task of motivational text generation with minimal computational resources.
*   **Custom-Built Dataset:** A significant aspect of this project is the **entirely custom-created dataset**, meticulously developed from scratch by the project owner. This bespoke dataset is crucial for training the model to understand and generate contextually relevant and effective motivational nudges.

## Project Context

This project was developed as a **graduation project**, showcasing advanced skills in Natural Language Processing (NLP), machine learning model fine-tuning, and data engineering. The emphasis on a custom dataset highlights a deep understanding of data preparation and its critical role in AI model performance.

## Getting Started

To set up and run this project locally, follow these steps:

### Prerequisites

Ensure you have Python 3.8+ installed. All necessary Python libraries are listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/FLAN-T5-Motivation-Model.git
    cd FLAN-T5-Motivation-Model
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Usage

The core logic for training and inference is contained within the `FLAN_T5.ipynb` Jupyter Notebook. This notebook demonstrates:

*   Data loading and preprocessing from `nudge_balanced_740.csv`.
*   Configuration and application of LoRA for fine-tuning the FLAN-T5 model.
*   Training the model.
*   Generating motivational nudges based on user input and mood.

### Running the Streamlit App

To run the interactive web application:

```bash
streamlit run app.py
```

This will open a local web server, and you can access the application in your browser.

### Running the Jupyter Notebook

The core logic for training and inference is contained within the `FLAN_T5.ipynb` Jupyter Notebook. This notebook demonstrates:

*   Data loading and preprocessing from `nudge_balanced_740.csv`.
*   Configuration and application of LoRA for fine-tuning the FLAN-T5 model.
*   Training the model.
*   Generating motivational nudges based on user input and mood.

To run the notebook:

```bash
jupyter notebook FLAN_T5.ipynb
```

## Files in this Repository

*   `app.py`: The Streamlit web application for interacting with the FLAN-T5 Motivation Model.
*   `FLAN_T5.ipynb`: The main Jupyter Notebook containing the model training, fine-tuning, and inference code.
*   `flan_nudge_balanced_best.zip`: Compressed file containing the fine-tuned model weights.
*   `nudge_balanced_740.csv`: The custom-built dataset used for training the model.
*   `requirements.txt`: Lists all Python dependencies required to run the project.

## Contribution

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. (Note: A LICENSE file will be added if needed.)

## Contact

For any inquiries, please contact [Your Name/Email/LinkedIn Profile].
