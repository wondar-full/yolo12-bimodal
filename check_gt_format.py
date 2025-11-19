import json

gt_path = r'F:/CV/Paper/yoloDepth/datasets/VisDrone2019-DET-COCO/annotations/VisDrone2019-DET_val_coco.json'
gt_data = json.load(open(gt_path))

print(f"📊 Total GT images: {len(gt_data['images'])}")
print("\n🔍 Sample GT images:")
for img in gt_data['images'][:10]:
    print(f"   id={img['id']} (type:{type(img['id']).__name__}), file_name={img['file_name']}")

print("\n🔍 Looking for specific predictions image_ids:")
search_names = ['0000256_02173_d_0000030.jpg', '0000249_02468_d_0000008.jpg', '0000364_01765_d_0000782.jpg']
for name in search_names:
    found = [img for img in gt_data['images'] if img['file_name'] == name]
    if found:
        print(f"   ✅ {name} → id={found[0]['id']}")
    else:
        print(f"   ❌ {name} NOT FOUND!")
