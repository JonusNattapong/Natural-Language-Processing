"""
Script สำหรับลบ field "language" จากไฟล์ ThaiNER.jsonl
"""

import json
import os

def remove_language_field(input_file, output_file=None, backup=True):
    """
    ลบ field "language" จากทุกบรรทัดในไฟล์ JSONL

    Parameters:
    -----------
    input_file : str
        ชื่อไฟล์ input (ThaiNER.jsonl)
    output_file : str, optional
        ชื่อไฟล์ output (ถ้าไม่ระบุจะเขียนทับไฟล์เดิม)
    backup : bool, optional
        สร้างไฟล์สำรองก่อนแก้ไข (default: True)
    """

    # อ่านข้อมูลทั้งหมด
    print(f"กำลังอ่านไฟล์: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"พบข้อมูลทั้งหมด {len(lines)} รายการ")

    # สร้างไฟล์สำรอง
    if backup and output_file is None:
        backup_file = input_file.replace('.jsonl', '_backup_before_remove_language.jsonl')
        print(f"กำลังสร้างไฟล์สำรอง: {backup_file}")
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    # ลบ field "language"
    cleaned_data = []
    removed_count = 0

    for line in lines:
        line = line.strip()
        if not line:  # ข้ามบรรทัดว่าง
            continue

        try:
            data = json.loads(line)

            # ลบ field "language" ถ้ามี
            if 'language' in data:
                del data['language']
                removed_count += 1

            cleaned_data.append(json.dumps(data, ensure_ascii=False))

        except json.JSONDecodeError as e:
            print(f"⚠️  ข้าม: บรรทัดที่มีปัญหา (error: {e})")
            print(f"    เนื้อหา: {line[:100]}...")
            continue

    # เขียนไฟล์ใหม่
    output = output_file if output_file else input_file
    print(f"\nกำลังเขียนไฟล์: {output}")
    with open(output, 'w', encoding='utf-8') as f:
        for line in cleaned_data:
            f.write(line + '\n')

    print(f"\n✓ ลบ field 'language' สำเร็จ!")
    print(f"  - จำนวนรายการทั้งหมด: {len(cleaned_data)}")
    print(f"  - ลบ field 'language' ออก: {removed_count} รายการ")
    print(f"  - ไฟล์ผลลัพธ์: {output}")
    if backup and output_file is None:
        print(f"  - ไฟล์สำรอง: {backup_file}")


if __name__ == "__main__":
    # กำหนดชื่อไฟล์
    input_file = "ThaiNER.jsonl"

    # ตรวจสอบว่าไฟล์มีอยู่หรือไม่
    if not os.path.exists(input_file):
        print(f"❌ ไม่พบไฟล์: {input_file}")
        print(f"   กรุณาตรวจสอบว่าคุณอยู่ใน directory ที่ถูกต้อง")
        exit(1)

    # เรียกใช้ฟังก์ชัน
    remove_language_field(
        input_file=input_file,
        output_file=None,  # เขียนทับไฟล์เดิม (มีสำรอง)
        backup=True        # สร้างไฟล์สำรอง
    )

    print("\nคำแนะนำ:")
    print("- ถ้าต้องการเก็บไฟล์เดิมไว้ ให้เปลี่ยน output_file เป็นชื่อไฟล์ใหม่")
    print("- ถ้าไม่ต้องการสร้างไฟล์สำรอง ให้เปลี่ยน backup=False")