import streamlit as st
import requests
import base64
from PIL import Image
import io
import os
import json
import time

# Server configuration
SERVER_URL = "http://129.120.61.163:8000"

# Set page title and configuration
st.set_page_config(
    page_title="Human Detection System",
    layout="wide"
)

# Helper function to display base64 images
def display_base64_image(base64_string, caption="", width=150):
    try:
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        st.image(img, caption=caption, width=width)
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

# Helper function to check server status
def check_server_status():
    try:
        response = requests.get(f"{SERVER_URL}/", timeout=5)
        if response.status_code == 200:
            return True
        return False
    except:
        return False

# Helper function to handle API requests with retries
def make_api_request(endpoint, method="post", data=None, files=None, retries=3, timeout=30):
    for attempt in range(retries):
        try:
            if method.lower() == "post":
                response = requests.post(f"{SERVER_URL}/{endpoint}", data=data, files=files, timeout=timeout)
            else:
                response = requests.get(f"{SERVER_URL}/{endpoint}", params=data, timeout=timeout)
            
            return response
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                st.warning(f"Retrying request... ({attempt+1}/{retries})")
                time.sleep(2)  # Wait before retrying
            else:
                raise e

# Title
st.title("Human Detection System")

# Check server connectivity
server_status = check_server_status()
if not server_status:
    st.error("⚠️ Cannot connect to the server. Please check if the server is running.")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Video Processing", "Text Search", "Image Search", "Database"])

# Video Processing Tab
with tab1:
    # Upload video file
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        # Display video in a smaller size
        st.video(uploaded_file, start_time=0)
        
        # Process button
        if st.button("Process Video"):
            with st.spinner("Processing video. This may take some time..."):
                try:
                    # Prepare the file for upload
                    files = {"file": uploaded_file}
                    
                    # Make the API request with retry mechanism
                    response = make_api_request("detect-humans-video/", files=files, timeout=180)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Store result in session state
                        st.session_state['video_result'] = result
                        
                        # Display success message
                        st.success(f"Found {result.get('total_human_count', 0)} unique humans.")
                    else:
                        st.error(f"Error: Status code {response.status_code}. {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Display results if available
    if 'video_result' in st.session_state:
        result = st.session_state['video_result']
        
        st.write(f"**Total humans:** {result.get('total_human_count', 0)} | **Processed frames:** {result.get('processed_frames', 0)}/{result.get('total_frames', 0)}")
        
        # Display detected humans if available
        if 'humans' in result and len(result['humans']) > 0:
            # Create a grid of images - use 6 columns for smaller display
            cols = st.columns(6)
            for i, human in enumerate(result['humans']):
                with cols[i % 6]:
                    if 'image_base64' in human:
                        display_base64_image(human['image_base64'], f"#{human.get('human_id', i+1)}", width=100)
        elif 'message' in result:
            st.info(result['message'])

# Text Search Tab
with tab2:
    # Text search form
    text_query = st.text_input("Enter description:", placeholder="E.g. 'person in red shirt'")
    
    if st.button("Search by Text"):
        if text_query:
            with st.spinner("Searching..."):
                try:
                    # Make API request with retry mechanism
                    response = make_api_request("fetch-by-text/", data={"text_query": text_query})
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Store search results in session state
                        st.session_state['text_search_result'] = result
                        
                        # Display success message
                        st.success(f"Found {len(result.get('results', []))} matches.")
                    else:
                        st.error(f"Error: Status code {response.status_code}. {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a description to search.")
    
    # Display results if available
    if 'text_search_result' in st.session_state:
        result = st.session_state['text_search_result']
        
        if result.get('results', []) and len(result['results']) > 0:
            st.subheader(f"Results for: '{result.get('query', 'search')}'")
            
            # Create a grid of images with similarity scores - use 5 columns for smaller display
            cols = st.columns(5)
            for i, item in enumerate(result['results']):
                with cols[i % 5]:
                    if 'payload' in item and 'base64_image' in item['payload']:
                        display_base64_image(item['payload']['base64_image'], f"Score: {item.get('score', 0):.2f}", width=120)
        else:
            st.info("No matching results found.")

# Image Search Tab
with tab3:
    # Image upload
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="image_search")
    
    if uploaded_file is not None:
        # Display the uploaded image in a smaller size
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=200)
            
            # Search button
            if st.button("Search by Image"):
                with st.spinner("Searching..."):
                    try:
                        # Save the uploaded file temporarily
                        temp_file_path = "temp_upload.jpg"
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Send with proper multipart/form-data
                        with open(temp_file_path, 'rb') as image_file:
                            files = {'file': (os.path.basename(temp_file_path), image_file, 'image/jpeg')}
                            response = make_api_request("fetch-by-image/", files=files)
                        
                        # Clean up temp file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Store search results in session state
                            st.session_state['image_search_result'] = result
                            
                            # Display success message
                            st.success(f"Found {len(result.get('results', []))} matches.")
                        else:
                            st.error(f"Error: Status code {response.status_code}. {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"Error opening image: {str(e)}")
    
    # Display results if available
    if 'image_search_result' in st.session_state:
        result = st.session_state['image_search_result']
        
        if result.get('results', []) and len(result['results']) > 0:
            st.subheader(f"Results for uploaded image")
            
            # Create a grid of images with similarity scores - use 5 columns for smaller display
            cols = st.columns(5)
            for i, item in enumerate(result['results']):
                with cols[i % 5]:
                    if 'payload' in item and 'base64_image' in item['payload']:
                        display_base64_image(item['payload']['base64_image'], f"Score: {item.get('score', 0):.2f}", width=120)
        else:
            st.info("No matching results found.")

# Database Tab
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Delete Collection")
        confirm_delete = st.checkbox("I understand this will delete all data")
        if st.button("Delete All Data", disabled=not confirm_delete):
            try:
                response = make_api_request("delete-collection/", method="get")
                if response.status_code == 200:
                    st.success("Database reset successfully!")
                else:
                    st.error(f"Error: Status code {response.status_code}. {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("Change Collection")
        new_name = st.text_input("New collection name", "images")
        if st.button("Update Collection"):
            if new_name:
                try:
                    response = make_api_request("change-collection/", method="get", data={"new_name": new_name})
                    if response.status_code == 200:
                        st.success(f"Collection changed to '{new_name}'")
                    else:
                        st.error(f"Error: Status code {response.status_code}. {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a collection name.")

# Footer
st.caption("Human Detection System • Powered by YOLOv3 and Fashion CLIP")