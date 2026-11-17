import cv2
import numpy as np
import streamlit as st
from datetime import datetime
from insightface.app import FaceAnalysis

st.set_page_config(
    page_title="Face Attendance System",
    page_icon="🧑‍🏫",
    layout="wide",
)

st.title("🧑‍🏫 Real-Time Face Attendance System")
st.caption("Capture a face, extract an ArcFace embedding, and mark attendance.")

@st.cache_resource
def load_face_engine():
    analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return analyzer

face_engine = load_face_engine()

if "gallery" not in st.session_state:
    st.session_state.gallery = {}

if "attendance" not in st.session_state:
    st.session_state.attendance = []

def get_embedding(image):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    faces = face_engine.get(frame)

    if not faces:
        return None, None

    face = max(faces, key=lambda f: f.det_score)
    return face.normed_embedding.astype(np.float32), face

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

tab_enroll, tab_mark, tab_logs = st.tabs(
    ["Enroll Person", "Mark Attendance", "Attendance Log"]
)

with tab_enroll:
    st.subheader("Enroll a person")

    person_name = st.text_input("Full name", key="enroll_name")
    reference_image = st.camera_input("Take enrollment photo", key="enroll_photo")

    if st.button("Enroll person"):
        if not person_name or reference_image is None:
            st.error("Enter a name and capture a reference photo.")
        else:
            embedding, face = get_embedding(reference_image)

            if embedding is None:
                st.error("No clear face detected. Try again with better lighting.")
            else:
                st.session_state.gallery[person_name] = embedding
                st.success(f"{person_name} enrolled successfully.")

with tab_mark:
    st.subheader("Mark attendance")

    course_session = st.text_input(
        "Course / session name",
        value="Default Session",
    )

    attendance_image = st.camera_input("Take attendance photo", key="attendance_photo")

    if st.button("Recognize and mark attendance"):
        if not st.session_state.gallery:
            st.error("No people are enrolled yet.")
        elif attendance_image is None:
            st.error("Capture an attendance photo first.")
        else:
            embedding, face = get_embedding(attendance_image)

            if embedding is None:
                st.error("No clear face detected.")
            else:
                best_name = None
                best_score = -1.0

                for name, stored_embedding in st.session_state.gallery.items():
                    score = cosine_similarity(embedding, stored_embedding)

                    if score > best_score:
                        best_name = name
                        best_score = score

                threshold = 0.38

                if best_score >= threshold:
                    already_marked = any(
                        record["person"] == best_name
                        and record["session"] == course_session
                        for record in st.session_state.attendance
                    )

                    if already_marked:
                        st.warning(
                            f"{best_name} has already been marked present "
                            f"for {course_session}."
                        )
                    else:
                        st.session_state.attendance.append(
                            {
                                "person": best_name,
                                "session": course_session,
                                "timestamp": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "confidence": round(best_score, 4),
                                "status": "Present",
                            }
                        )

                        st.success(
                            f"Attendance marked for {best_name} "
                            f"with confidence {best_score:.4f}."
                        )
                else:
                    st.error(
                        f"Unknown person. Best match: {best_name} "
                        f"with confidence {best_score:.4f}."
                    )

with tab_logs:
    st.subheader("Attendance records")

    if st.session_state.attendance:
        st.dataframe(
            st.session_state.attendance,
            use_container_width=True,
        )
    else:
        st.info("No attendance has been recorded yet.")

    if st.session_state.attendance:
        csv_data = "Person,Session,Timestamp,Confidence,Status\n"

        for record in st.session_state.attendance:
            csv_data += (
                f"{record['person']},{record['session']},"
                f"{record['timestamp']},{record['confidence']},"
                f"{record['status']}\n"
            )

        st.download_button(
            "Download attendance CSV",
            data=csv_data,
            file_name="attendance.csv",
            mime="text/csv",
        )
