import streamlit as st
import torch
from utilities import preprocess_dna_sequence, count_cpg
from model import CpGPredictor

# Load the pre-trained model
def load_model(model_path: str):
    model = CpGPredictor(input_size=6, hidden_size=256, num_layers=2, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Streamlit web app
def main():
    # Load the model
    model_path = 'best_model_lstm.pt'
    model = load_model(model_path)

    st.title('CpG Detector')
    st.write("This app detects the number of CpGs (consecutive 'CG' patterns) in a given DNA sequence using a neural network model.")

    # Input text box for DNA sequence
    sequence = st.text_input("Enter a DNA sequence:", "NCACANNTNCGGAGGCGNA")

    # Button to trigger prediction
    if st.button("Detect CpGs"):
        # Preprocess sequence for model
        input_tensor = preprocess_dna_sequence(sequence)
        lengths = torch.tensor([input_tensor.shape[1]])  # Get length for the current sequence

        # Get model prediction
        with torch.no_grad():
            model_output = model(input_tensor, lengths).item()

        # Display the outputs
        st.write(f"Raw Model Output: {model_output}")
        st.write(f"Corrected Output (Rounded CpG Count): {round(model_output)}")

        # Count CpG using regex for reference
        actual_cpg_count = count_cpg(sequence)
        st.write(f"Actual CpG Count (by regex): {actual_cpg_count}")

if __name__ == '__main__':
    main()
