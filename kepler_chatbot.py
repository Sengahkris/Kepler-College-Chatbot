import pandas as pd
import streamlit as st
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import hashlib

# Set page config
st.set_page_config(page_title="Kepler Thinking Chatbot", page_icon="🧠")

# Load pre-trained semantic model (free alternative to OpenAI)
@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Enhanced data loader with semantic embeddings
@st.cache_data
def load_data():
    excel_file = "kepler_data.xlsx"
    sheets = {
        "Draft": None,
        "Admissions": None,
        "Orientation": None,
        "Programs": None
    }
    
    try:
        model = load_semantic_model()
        knowledge_graph = defaultdict(list)
        
        for sheet in sheets.keys():
            df = pd.read_excel(excel_file, sheet_name=sheet)
            df.columns = ["Questions", "Answers"]
            
            # Generate embeddings for all questions
            questions = df["Questions"].astype(str).tolist()
            embeddings = model.encode(questions)
            
            # Store in knowledge graph
            for idx, row in df.iterrows():
                entry = {
                    "question": str(row["Questions"]),
                    "answer": str(row["Answers"]),
                    "embedding": embeddings[idx],
                    "source": sheet,
                    "related": []
                }
                knowledge_graph[sheet].append(entry)
            
            sheets[sheet] = df
        
        # Build relationships between concepts
        build_relationships(knowledge_graph)
        return sheets, knowledge_graph
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def build_relationships(graph):
    """Create connections between related concepts across different sheets"""
    all_entries = []
    for sheet in graph.values():
        all_entries.extend(sheet)
    
    for i, entry1 in enumerate(all_entries):
        for j, entry2 in enumerate(all_entries[i+1:], i+1):
            # Calculate semantic similarity
            sim = cosine_similarity(
                [entry1["embedding"]],
                [entry2["embedding"]]
            )[0][0]
            
            # If highly related, connect them
            if sim > 0.7:  # Adjust threshold as needed
                entry1["related"].append((j, sim))
                entry2["related"].append((i, sim))

# Semantic search with reasoning
def semantic_search(user_question, knowledge_graph, model):
    # Encode user question
    question_embedding = model.encode([user_question])
    
    best_match = None
    best_score = 0
    best_source = None
    reasoning = []
    
    # Search across all knowledge
    for sheet, entries in knowledge_graph.items():
        for entry in entries:
            # Calculate semantic similarity
            sim = cosine_similarity(
                [question_embedding[0]],
                [entry["embedding"]]
            )[0][0]
            
            # If this is the best match so far
            if sim > best_score:
                best_score = sim
                best_match = entry
                best_source = sheet
                reasoning = [f"Matched to: '{entry['question']}' (similarity: {sim:.2f})"]
                
                # Check related concepts
                for rel_idx, rel_sim in entry["related"]:
                    rel_entry = None
                    # Find the related entry in the graph
                    for s in knowledge_graph.values():
                        if rel_idx < len(s):
                            rel_entry = s[rel_idx]
                            break
                        rel_idx -= len(s)
                    
                    if rel_entry and rel_sim > 0.6:
                        reasoning.append(f"Related concept: '{rel_entry['question']}' (similarity: {rel_sim:.2f})")
    
    # Dynamic threshold based on question length
    threshold = max(0.5, 0.7 - (0.02 * len(user_question.split())))
    
    if best_score > threshold:
        # Build comprehensive answer
        answer = best_match["answer"]
        
        # Add related information if relevant
        related_info = []
        for rel_idx, rel_sim in best_match["related"]:
            if rel_sim > 0.6:
                rel_entry = None
                for s in knowledge_graph.values():
                    if rel_idx < len(s):
                        rel_entry = s[rel_idx]
                        break
                    rel_idx -= len(s)
                
                if rel_entry:
                    related_info.append(f"\n\nRelated info: {rel_entry['answer']}")
        
        if related_info:
            answer += "\n\n" + "\n".join(related_info[:2])  # Include max 2 related items
        
        return answer, best_source, reasoning
    
    return None, None, []

# Learning mechanism
def learn_from_interaction(user_question, response, knowledge_graph, model):
    """Store new information from successful interactions"""
    # Only learn if we provided a good answer
    if response and len(response) > 20:  # Only learn substantial answers
        # Create hash as simple ID
        q_hash = hashlib.md5(user_question.encode()).hexdigest()[:8]
        
        # Add to knowledge graph
        new_entry = {
            "question": user_question,
            "answer": response,
            "embedding": model.encode([user_question])[0],
            "source": "learned",
            "related": []
        }
        
        if "learned" not in knowledge_graph:
            knowledge_graph["learned"] = []
        knowledge_graph["learned"].append(new_entry)
        
        # Rebuild relationships for this new entry
        build_relationships(knowledge_graph)

# Main app function
def main():
    st.title("Kepler Thinking Assistant 🧠")
    st.write("Ask me anything - I understand context and make connections!")
    
    # Load data and model
    data, knowledge_graph = load_data()
    model = load_semantic_model()
    
    if data is None:
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.knowledge_graph = knowledge_graph
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("What would you like to know about Kepler?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response with reasoning
        response, source, reasoning = semantic_search(prompt, st.session_state.knowledge_graph, model)
        
        if response:
            # Learn from this interaction
            learn_from_interaction(prompt, response, st.session_state.knowledge_graph, model)
            
            # Format response
            formatted_response = f"{response}\n\n*(Source: {source} section)*"
            
            # Add reasoning in expander
            with st.expander("How I arrived at this answer"):
                st.write("\n".join(reasoning))
        else:
            formatted_response = ("I'm not entirely sure about that. Based on what I know, "
                               "you might want to ask about:\n\n"
                               "- Admission requirements\n"
                               "- Program details\n"
                               "- Orientation schedules\n\n"
                               "Could you rephrase or ask about one of these areas?")
        
        # Display bot response
        with st.chat_message("assistant"):
            st.markdown(formatted_response)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})

if __name__ == "__main__":
    main()