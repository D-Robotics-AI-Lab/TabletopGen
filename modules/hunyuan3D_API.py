from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.ai3d.v20250513 import ai3d_client, models

import requests
import time
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
import json
import os

# Temporarily disable proxy settings
_proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
_old_proxy = {k: os.environ.get(k) for k in _proxy_keys}



def to_base64(path):
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

image_path = "path/to/your/image.jpg" 

def submit_job(client, image_path):
    # Construct request
    req = models.SubmitHunyuanTo3DProJobRequest()
    
    params = {
        "ImageBase64": to_base64(image_path),
        "GenerateType": "Normal",  # Normal/LowPoly/Geometry/Sketch
        "EnablePBR": True,  # Optional: Enable PBR material
        # "FaceCount": 500000,  # Optional: Model face count
    }
    req.from_json_string(json.dumps(params))

    # Call interface
    resp = client.SubmitHunyuanTo3DProJob(req)
    job_id = resp.JobId
    print(f"Job submitted, JobID: {job_id}")
    return job_id

def query_job(client, job_id):
    req = models.QueryHunyuanTo3DProJobRequest()
    req.JobId = job_id
    return client.QueryHunyuanTo3DProJob(req)


def gen_single_obj_hy3dapi(SecretId, SecretKey, image_path, output_glb_path):
    # Configure secret key and region
    cred = credential.Credential(SecretId, SecretKey)  # Replace with actual secret key
    http_profile = HttpProfile(endpoint="ai3d.tencentcloudapi.com")
    client_profile = ClientProfile(httpProfile=http_profile)
    client = ai3d_client.Ai3dClient(cred, "ap-guangzhou", client_profile)


    try:
        job_id = submit_job(client, image_path)
        print("Submission successful, JobId:", job_id)

        # Poll until completion
        while True:
            r = query_job(client, job_id)
            status = r.Status  # WAIT / RUN / FAIL / DONE
            print("Status:", status)
            if status == "DONE":
                # Get GLB from the returned 3D file list
                glb_url = None
                if r.ResultFile3Ds:
                    for f in r.ResultFile3Ds:
                        if (getattr(f, "Type", "") or "").upper() == "GLB":
                            glb_url = f.Url
                            break
                    # If no explicit Type, take the first one directly
                    if not glb_url:
                        glb_url = r.ResultFile3Ds[0].Url
                if not glb_url:
                    raise RuntimeError("Task completed but no GLB link returned.")

                # Download and save
                data = requests.get(glb_url, timeout=120).content
                with open(output_glb_path, "wb") as f:
                    f.write(data)
                print("Saved to:", output_glb_path)
                break

            if status == "FAIL":
                msg = getattr(r, "ErrorMessage", "Unknown error")
                code = getattr(r, "ErrorCode", "")
                raise RuntimeError(f"Task failed: {code} {msg}")

            time.sleep(15)  # Polling

    except TencentCloudSDKException as e:
        print("SDK Exception:", str(e))


if __name__ == "__main__":
    SecretId=""
    SecretKey=""
