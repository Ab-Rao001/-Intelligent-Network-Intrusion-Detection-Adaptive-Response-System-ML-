import os
import re

d = 'streamlit_app/pages'
for f in os.listdir(d):
    if not f.endswith('.py'): continue
    path = os.path.join(d, f)
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Check if from utils.theme import apply_custom_theme is present
    if 'apply_custom_theme' not in content:
        content = content.replace('st.set_page_config(', 'from utils.theme import apply_custom_theme\n\nst.set_page_config(')
    
    # Replace the existing style blocks
    # Using regex to find st.markdown("""<style> ... </style>""", unsafe_allow_html=True)
    pattern = r'st\.markdown\(\"\"\"\s*<style>.*?</style>\s*\"\"\",\s*unsafe_allow_html=True\)'
    content = re.sub(pattern, 'apply_custom_theme()', content, flags=re.DOTALL)
    
    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)
print("Styles replaced!")
