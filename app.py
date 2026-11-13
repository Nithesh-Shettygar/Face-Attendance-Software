import os
import re
import time
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime, date

import cv2
import numpy as np
import face_recognition
import mysql.connector
from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)

DATASET_DIR = "dataset"      # a copy of each registered face photo is kept here
TOLERANCE = 0.5              # lower = stricter matching
SCALE = 0.25                 # shrink frames for faster detection
PROCESS_EVERY_N = 2          # run recognition on every Nth frame
RECHECK_SECONDS = 20         # how often the same face may be written to the database
CONFIRM_FRAMES = 3           # face must be matched in this many consecutive checks before attendance is marked

# ---------- SAME database as the college portal ----------
# These must match DB_CONFIG / DB_NAME in the portal's app.py.
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', ''),
}
DB_NAME = 'college_portal'

os.makedirs(DATASET_DIR, exist_ok=True)


def get_db():
    return mysql.connector.connect(**DB_CONFIG, database=DB_NAME)


@contextmanager
def db_cursor(commit=False):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    try:
        yield cursor
        if commit:
            conn.commit()
    finally:
        cursor.close()
        conn.close()


def init_face_tables():
    """Creates the face system's own tables inside the portal database.
    The portal must have been started at least once so that `students` and `subjects` exist."""
    conn = get_db()
    cur = conn.cursor()
    # One row per student whose face is registered in the face attendance software
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_registrations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_id INT NOT NULL UNIQUE,
            registration_no VARCHAR(50) NOT NULL UNIQUE,
            image_file VARCHAR(255) NOT NULL,
            encoding BLOB NOT NULL,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_face_student FOREIGN KEY (student_id)
                REFERENCES students(id) ON DELETE CASCADE
        ) ENGINE=InnoDB""")
    # One row every time the camera marks someone Present (audit trail)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_attendance_log (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_id INT NOT NULL,
            subject_id INT NOT NULL,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_fal_student FOREIGN KEY (student_id)
                REFERENCES students(id) ON DELETE CASCADE,
            CONSTRAINT fk_fal_subject FOREIGN KEY (subject_id)
                REFERENCES subjects(id) ON DELETE CASCADE
        ) ENGINE=InnoDB""")
    conn.commit()
    cur.close()
    conn.close()


try:
    init_face_tables()
    print("[INFO] Connected to the college_portal database. Face tables are ready.")
except mysql.connector.Error as err:
    raise SystemExit(
        f"[ERROR] Cannot use the college_portal database: {err}\n"
        f"Start MySQL and run the college portal once (it creates the students/subjects tables), "
        f"then start this app again.")


# ---------- Timetable helpers (same rules as the portal) ----------
def slot_range(label):
    """'09:00 AM - 10:00 AM' -> (start_minute, end_minute) or None."""
    times = re.findall(r'(\d{1,2}):(\d{2})\s*(AM|PM)', label or '', re.I)
    if len(times) < 2:
        return None

    def to_min(t):
        h = int(t[0]) % 12
        if t[2].upper() == 'PM':
            h += 12
        return h * 60 + int(t[1])

    return to_min(times[0]), to_min(times[1])


def open_class(cur, course, semester):
    """The subject class running right now for this course + semester, else None."""
    now = datetime.now()
    minutes = now.hour * 60 + now.minute
    cur.execute("""
        SELECT e.subject_id, t.time_label, s.subject_code, s.subject_name
        FROM timetable_entries e
        JOIN timetable_slots t ON t.id = e.slot_id
        JOIN subjects s ON s.id = e.subject_id
        WHERE t.course_code = %s AND t.semester = %s
          AND e.entry_type = 'subject' AND e.day_name = %s
    """, (course, semester, now.strftime('%A')))
    for r in cur.fetchall():
        rng = slot_range(r['time_label'])
        if rng and rng[0] <= minutes < rng[1]:
            return r
    return None


# ---------- Marking attendance directly in the portal's attendance table ----------
def mark_present(student_id):
    """Returns {'status': marked|already|no_class|unknown, 'message': ...}"""
    with db_cursor(commit=True) as cur:
        cur.execute("SELECT id, course, semester FROM students WHERE id = %s", (student_id,))
        stu = cur.fetchone()
        if not stu:
            return {"status": "unknown", "message": "Student is no longer in the portal"}

        cls = open_class(cur, stu['course'], stu['semester'])
        if not cls:
            return {"status": "no_class",
                    "message": f"No class running now for {stu['course']} Sem {stu['semester']}"}

        label = f"{cls['subject_code']} - {cls['subject_name']}"
        today = date.today()

        cur.execute("""SELECT status FROM attendance
                       WHERE student_id=%s AND subject_id=%s AND att_date=%s""",
                    (student_id, cls['subject_id'], today))
        row = cur.fetchone()
        if row and row['status'] == 'Present':
            return {"status": "already", "message": label}

        cur.execute("SELECT faculty_id FROM faculty_assignments WHERE subject_id=%s",
                    (cls['subject_id'],))
        fa = cur.fetchone()
        fid = fa['faculty_id'] if fa else None

        cur.execute("""
            INSERT INTO attendance (student_id, subject_id, faculty_id, att_date, status)
            VALUES (%s,%s,%s,%s,'Present')
            ON DUPLICATE KEY UPDATE status='Present',
                                    faculty_id=COALESCE(VALUES(faculty_id), faculty_id)
        """, (student_id, cls['subject_id'], fid, today))
        cur.execute("INSERT INTO face_attendance_log (student_id, subject_id) VALUES (%s,%s)",
                    (student_id, cls['subject_id']))
        return {"status": "marked", "message": label}


# ---------- Popup events (polled by the browser) ----------
events = []
event_counter = 0
events_lock = threading.Lock()


def push_event(kind, roll_no, name, message=""):
    global event_counter
    with events_lock:
        event_counter += 1
        events.append({
            "id": event_counter,
            "type": kind,   # captured | already | noclass | unknown | offline
            "roll_no": roll_no,
            "name": name,
            "message": message,
            "time": datetime.now().strftime("%H:%M:%S"),
        })
        del events[:-50]


last_sent = {}      # student_id -> last time this face was written to the database
streaks = {}        # student_id -> consecutive recognised checks
face_state = {}     # student_id -> "captured" / "already" / "noclass" ... (shown on the video box)


def handle_detection(student_id, reg_no, name):
    """Runs in a background thread so the video never freezes."""
    try:
        res = mark_present(student_id)
    except mysql.connector.Error as err:
        print(f"[ERROR] attendance update failed: {err}")
        push_event("offline", reg_no, name, "Database error")
        return
    kind = {"marked": "captured", "already": "already",
            "no_class": "noclass", "unknown": "unknown"}.get(res["status"], "offline")
    face_state[student_id] = kind
    push_event(kind, reg_no, name, res.get("message", ""))


# ---------- Camera (single background reader, shared safely) ----------
class Camera:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.lock = threading.Lock()
        self.frame = None
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while True:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.05)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


camera = Camera(0)


# ---------- Registered faces (stored in the face_registrations table) ----------
faces_lock = threading.Lock()
known_encodings = []
known_metadata = []   # {"student_id", "roll_no", "name"}


def clean_roll(s):
    return re.sub(r"[^A-Za-z0-9-]", "", s)


def clean_name(s):
    return re.sub(r"[^A-Za-z ]", "", s).strip()


def load_registered_faces():
    """(Re)load all registered faces from the database."""
    with db_cursor() as cur:
        cur.execute("""
            SELECT f.student_id, f.registration_no, f.encoding, s.full_name
            FROM face_registrations f JOIN students s ON s.id = f.student_id
        """)
        rows = cur.fetchall()
    encodings = [np.frombuffer(bytes(r['encoding']), dtype=np.float64) for r in rows]
    metadata = [{"student_id": r['student_id'], "roll_no": r['registration_no'],
                 "name": r['full_name']} for r in rows]
    with faces_lock:
        known_encodings[:] = encodings
        known_metadata[:] = metadata
    print(f"[INFO] {len(rows)} registered face(s) loaded from the database.")


def import_existing_dataset():
    """One-time convenience: photos already in dataset/ (named rollno__name.jpg) are added to the
    face_registrations table if that registration number exists in the portal."""
    imported = 0
    for filename in os.listdir(DATASET_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")) or "__" not in filename:
            continue
        roll = os.path.splitext(filename)[0].split("__", 1)[0]
        try:
            with db_cursor(commit=True) as cur:
                cur.execute("SELECT id, registration_no FROM students WHERE registration_no=%s", (roll,))
                stu = cur.fetchone()
                if not stu:
                    continue
                cur.execute("SELECT id FROM face_registrations WHERE student_id=%s", (stu['id'],))
                if cur.fetchone():
                    continue
                img = face_recognition.load_image_file(os.path.join(DATASET_DIR, filename))
                encs = face_recognition.face_encodings(img)
                if not encs:
                    continue
                cur.execute("""INSERT INTO face_registrations (student_id, registration_no, image_file, encoding)
                               VALUES (%s,%s,%s,%s)""",
                            (stu['id'], stu['registration_no'], filename, encs[0].tobytes()))
                imported += 1
        except mysql.connector.Error as err:
            print(f"[WARN] could not import {filename}: {err}")
    if imported:
        print(f"[INFO] Imported {imported} existing photo(s) from {DATASET_DIR}/ into the database.")


import_existing_dataset()
load_registered_faces()


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register")
def register_page():
    return render_template("register.html")


def generate_raw_stream():
    """Plain camera preview (no recognition) used on the register page."""
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        time.sleep(0.03)


@app.route("/raw_feed")
def raw_feed():
    return Response(generate_raw_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/events")
def get_events():
    """Browser polls this. Without ?after= it just returns the latest id."""
    after = request.args.get("after", type=int)
    with events_lock:
        if after is None:
            return jsonify({"last_id": event_counter, "events": []})
        new = [e for e in events if e["id"] > after]
        return jsonify({"last_id": event_counter, "events": new})


@app.route("/api/registrations")
def get_registrations():
    """How many students are registered in the face attendance software."""
    try:
        with db_cursor() as cur:
            cur.execute("""
                SELECT f.registration_no, s.full_name, s.course_sem, f.registered_at
                FROM face_registrations f JOIN students s ON s.id = f.student_id
                ORDER BY f.registered_at DESC
            """)
            rows = cur.fetchall()
            cur.execute("SELECT COUNT(*) AS n FROM students")
            total_students = int(cur.fetchone()['n'])
        for r in rows:
            r['registered_at'] = r['registered_at'].strftime('%Y-%m-%d %H:%M')
        return jsonify({"success": True, "count": len(rows),
                        "total_students": total_students, "registrations": rows})
    except mysql.connector.Error as err:
        return jsonify({"success": False, "message": f"Database error: {err}"})


@app.route("/api/student/<roll_no>")
def lookup_student(roll_no):
    """Used by the register page to auto-fill the name from the portal."""
    roll_no = clean_roll(roll_no.strip())
    try:
        with db_cursor() as cur:
            cur.execute("SELECT registration_no, full_name FROM students WHERE registration_no = %s",
                        (roll_no,))
            stu = cur.fetchone()
        if not stu:
            return jsonify({"success": False, "message": "Not found in the college portal."})
        return jsonify({"success": True, "name": stu['full_name']})
    except mysql.connector.Error as err:
        return jsonify({"success": False, "message": f"Database error: {err}"})


def same_name(a, b):
    norm = lambda x: re.sub(r"[^a-z]", "", (x or "").lower())
    return norm(a) == norm(b)


@app.route("/api/register", methods=["POST"])
def register_face():
    roll_no = clean_roll(request.form.get("roll_no", "").strip())
    typed_name = request.form.get("name", "").strip()
    if not roll_no:
        return jsonify({"success": False, "message": "A valid Registration Number is required."})

    # Cross-check the registration number against the portal's students table
    try:
        with db_cursor() as cur:
            cur.execute("""SELECT id, registration_no, full_name FROM students
                           WHERE registration_no = %s""", (roll_no,))
            stu = cur.fetchone()
            if not stu:
                return jsonify({"success": False,
                                "message": f"{roll_no} is not a student in the college portal. "
                                           f"Add the student in the portal first."})
            if typed_name and not same_name(typed_name, stu['full_name']):
                return jsonify({"success": False,
                                "message": f"Name does not match the portal record for {roll_no} "
                                           f"({stu['full_name']}). Leave it blank or correct it."})
            cur.execute("SELECT id FROM face_registrations WHERE student_id = %s", (stu['id'],))
            if cur.fetchone():
                return jsonify({"success": False,
                                "message": f"{stu['registration_no']} already has a face registered."})
    except mysql.connector.Error as err:
        return jsonify({"success": False, "message": f"Database error: {err}"})

    frame = camera.get_frame()
    if frame is None:
        return jsonify({"success": False, "message": "Camera not available."})

    # Make sure exactly one face is visible before saving
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    if len(locations) == 0:
        return jsonify({"success": False, "message": "No face detected. Look at the camera and try again."})
    if len(locations) > 1:
        return jsonify({"success": False, "message": "Multiple faces detected. Only one person please."})

    encoding = face_recognition.face_encodings(rgb, locations)[0]

    reg_no = stu['registration_no']
    safe_name = clean_name(stu['full_name']).lower() or reg_no.lower()
    image_file = f"{clean_roll(reg_no)}__{safe_name}.jpg"
    cv2.imwrite(os.path.join(DATASET_DIR, image_file), frame)

    try:
        with db_cursor(commit=True) as cur:
            cur.execute("""INSERT INTO face_registrations (student_id, registration_no, image_file, encoding)
                           VALUES (%s,%s,%s,%s)""",
                        (stu['id'], reg_no, image_file, encoding.tobytes()))
    except mysql.connector.Error as err:
        return jsonify({"success": False, "message": f"Database error: {err}"})

    load_registered_faces()   # so the live scanner recognises them immediately
    return jsonify({"success": True,
                    "message": f"Registered {stu['full_name']} ({reg_no}) from the college portal!"})


def generate_attendance_stream():
    frame_count = 0
    last_results = []

    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        if frame_count % PROCESS_EVERY_N == 0:
            try:
                small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(rgb_small)
                encodings = face_recognition.face_encodings(rgb_small, locations)

                with faces_lock:
                    enc_list = list(known_encodings)
                    meta_list = list(known_metadata)

                results = []
                seen = set()
                for enc, (top, right, bottom, left) in zip(encodings, locations):
                    label, color = "Unknown", (0, 0, 255)

                    if enc_list:
                        distances = face_recognition.face_distance(enc_list, enc)
                        best = int(np.argmin(distances))
                        if distances[best] < TOLERANCE:
                            s = meta_list[best]
                            sid = s["student_id"]
                            seen.add(sid)
                            streaks[sid] = streaks.get(sid, 0) + 1

                            label = f"{s['roll_no']} - {s['name']}"
                            color = (0, 255, 0)

                            # only act once the face is confirmed over several checks
                            if streaks[sid] >= CONFIRM_FRAMES:
                                if face_state.get(sid) in ("captured", "already"):
                                    label += "  [PRESENT]"
                                now = time.time()
                                if now - last_sent.get(sid, 0) > RECHECK_SECONDS:
                                    last_sent[sid] = now
                                    threading.Thread(target=handle_detection,
                                                     args=(sid, s["roll_no"], s["name"]),
                                                     daemon=True).start()
                            else:
                                label += "  (verifying...)"

                    box = tuple(int(v / SCALE) for v in (top, right, bottom, left))
                    results.append((box, label, color))

                # reset streaks for anyone who left the frame
                for sid in list(streaks):
                    if sid not in seen:
                        del streaks[sid]
                last_results = results
            except Exception:
                traceback.print_exc()   # shows the real error in the terminal
                last_results = []

        frame_count += 1

        for (top, right, bottom, left), label, color in last_results:
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, max(top - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(generate_attendance_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    # debug/reloader off: the reloader would open the camera twice
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=True)