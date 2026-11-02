# axis-face

A self contained docker based facial recognition application for Axis cameras.
It could also be used with any IP camera that can send image snapshots via MQTT.  Initially built with Nvidia GPU acceleration but found that CPU is good enough, adding only 10ms to detections.

I vibe coded this in about 4 hours using Google Gemini Argon 4 with medium thinking.  Sorry, there are hardly any comments as I didn't tell Argon to include them.  My initial prompt:

<code>
I want help building a Docker based facial recognition server that will take cropped face images produced by Axis cameras that send images over MQTT. This will be running on a Windows 11 machine with Docker Desktop. It will use an Nvidia RTX 5090 for GPU acceleration. It should have a simple web UI that a user can associate names to detected faces and store all unknown faces. It should keep a historical record of when, which camera, and who was seen. It should be able to trigger email alerts when either known or unknown faces are seen. Keep only the best image(s) of a given face each hour to avoid using too much storage. Let's start by giving me a high level development plan to review and then we will walk through each step slowly as we build as I may have questions or clarifications as we go.
</code> 

Key features:
- Switchable CPU or GPU acceleration for InsightFace vision model
- Persistent SQLite DB in the volume mount
- Admin or guest login
- Merge function in case a new detection doesn't categorize to an existing detection (slower cameras may track slow and give you a bad crop)
- Custom tagging for categorization/gropping and alerting
- E-mail alerting based on known, unknown, or custom tag
- Can set friendly names for cameras (Axis defaults to using MAC addresses for naming)

./axis-face-dashboard.png

How to build and run it:
- Pre-reqs
  - Docker
  - An Axis IP camera with an integrated deep learning processing unit (DLPU) setup to send com.axis.scene.object_snapshot.v1#1 via MQTT.  I've tested with ARTPEC-8, ARTPEC-9, and Ambarella CV25 SoC's
  - (optional) Nvidia GPU; really overkill for this but if you plan on extending this further into complex image recognition, it might become helpful.
- `git clone https://github.com/hasherati/axis-face.git`
- cp .env.example .env and update .env per your environment
- `docker compose up -d`
- point browser to http://localhost:8000/

Here are all the files and what they do:
```
D:\docker\axis-face\
├── docker-compose.yaml         # will build the MQTT server and face-server without GPU acceleration
├── docker-compose.gpu.yaml     # if you really need GPU acceleration, build with this instead and set .env EXECUTION_MODE
├── Dockerfile
├── requirements.txt            # change versions at your own risk, some have version interdependencies
├── .env                        # modify per your environment
├── mosquitto\
│   ├── config\mosquitto.conf   # Listener 1883, allow_anonymous true, max_packet_size 20MB
│   ├── data\
│   └── log\
├── data\
│   ├── db\faces.db             # SQLite database (WAL mode)
│   ├── images\                 # Stored face crops: person_<id>/<YYYY-MM-DD_HH>/<filename>.jpg
│   └── models\                 # Bind-mounted to /root/.insightface in container
└── app\
    ├── main.py                 # FastAPI web server, UI routes, test endpoints
    ├── database.py             # SQLite schema, Top-3 hourly image pruning, merge/delete helpers
    ├── face_engine.py          # InsightFace GPU wrapper, 5-point geometry gate, quality scoring
    ├── mqtt_worker.py          # Async MQTT subscriber for Axis scene/object_snapshot/v1/1
    ├── email_alerter.py        # Async SMTP HTML email alerts with inline image & cooldowns
    └── templates\
        ├── index.html          # Tailwind CSS Web UI (Faces table, History, Settings)
        └── login.html          # Initial login page
```

Security
- As you can see, there is NO HTTPS so you should definitely put this behind an HTTPS proxy.
- 