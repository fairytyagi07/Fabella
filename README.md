# Fabella

**Where drawings magically come to life!**

Fabella is an AI-powered interactive storytelling platform designed for children.  
It allows users to draw simple sketches, recognizes those drawings using a deep learning model, and transforms them into personalized stories.

The project encourages children to become active participants instead of passive consumers of digital content by combining **drawing, imagination, storytelling, and audio narration**.

---

## About the Project

Children today spend a significant amount of time consuming digital content, but much of this interaction is passive.

Fabella aims to make screen time more creative and engaging.

Children can:

- Draw their own objects
- Have their drawings recognized by AI
- Generate a story containing those objects
- Choose the genre and length of the story
- Listen to the generated story using audio narration
- Save their stories for later

---

## Features

### Drawing Interface
Children can freely create sketches using different colours and brush sizes.

### AI Sketch Recognition
The drawings are recognized using a **Convolutional Neural Network (CNN)** trained using the **Quick, Draw! dataset**.

### AI Story Generation
The objects identified from the drawings are passed to **Llama 3 through Ollama**, which generates a story incorporating the recognized objects.

### Audio Narration
Generated stories can also be converted into speech using **pyttsx3**, allowing children to listen to their stories.

### Parent & Child Profiles
Parents can log into the application, after which children can select their name and avatar.

### Story Storage
User information, drawings, and stories can be stored using **MongoDB**.

### Child-Friendly UI
The application provides a colourful and simple interface designed specifically for children.

---

## How Fabella Works

1. A parent creates or logs into an account.
2. The child selects their profile and avatar.
3. The child creates **three drawings**.
4. The sketches are passed to the trained CNN model.
5. The CNN predicts the objects represented by the drawings.
6. The three predicted words are passed to **Ollama / Llama 3**.
7. Llama 3 generates a story containing the predicted objects.
8. The generated story is displayed to the user.
9. **pyttsx3** converts the generated story into audio.
10. The child can read or listen to the generated story.

---

## System Architecture

```text
             Child Drawings
                   │
                   ▼
            CNN Recognition
                   │
                   ▼
          Predicted Objects
                   │
                   ▼
             Ollama / Llama 3
                   │
                   ▼
          Generated Story
              │        │
              │        ▼
              │    pyttsx3
              │        │
              ▼        ▼
           Text      Audio
              │
              ▼
            Fabella
