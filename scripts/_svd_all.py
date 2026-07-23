import requests

r = requests.get("https://zenodo.org/api/records/16874898", timeout=15)
data = r.json()
files = data["files"]
files.sort(key=lambda x: x["size"])
print(f"Total files: {len(files)}")
print(f"Total size: {sum(f['size'] for f in files)//1024//1024}MB")
print()
for f in files:
    print(f"  {f['size']//1024//1024:4d}MB  {f['key']}")
