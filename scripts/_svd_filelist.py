import requests

r = requests.get("https://zenodo.org/api/records/16874898", timeout=15)
data = r.json()
files = data["files"]

normal_files = [f for f in files if "Normal" in f["key"] or "normal" in f["key"].lower()]
print("Normal/healthy files:")
for f in normal_files:
    print(f"  {f['key']}: {f['size']//1024//1024}MB  url={f['links']['self'][:70]}")

print()
path_files = [f for f in files if "Normal" not in f["key"]]
path_files.sort(key=lambda x: x["size"])
print("Smallest pathological (first 10):")
for f in path_files[:10]:
    print(f"  {f['key']}: {f['size']//1024//1024}MB")
