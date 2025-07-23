# run.py
# import uvicorn

# if __name__ == "__main__":
#     uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
import os
import uvicorn
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render sets this env var
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)

    