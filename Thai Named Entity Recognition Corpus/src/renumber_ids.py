"""
Script สำหรับเรียง ID ใหม่ในไฟล์ ThaiNER.jsonl
จะเรียงลำดับ ID เป็น thner_0001, thner_0002, ... ตามลำดับบรรทัด
"""

import json
import os

def renumber_ids(input_file, output_file=None, backup=True):
    """
    เรียง ID ใหม่ในไฟล์ JSONL
    
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
        backup_file = input_file.replace('.jsonl', '_backup.jsonl')
        print(f"กำลังสร้างไฟล์สำรอง: {backup_file}")
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    # เรียง ID ใหม่
    renumbered_data = []
    counter = 0  # นับเฉพาะบรรทัดที่มีข้อมูล
    domain_count = {}  # นับจำนวน domain
    
    for line in lines:
        line = line.strip()
        if not line:  # ข้ามบรรทัดว่าง
            continue
        
        try:
            data = json.loads(line)
            counter += 1
            
            # เปลี่ยน ID เป็น thner_xxxx (เช่น thner_0001, thner_0002)
            new_id = f"thner_{counter:04d}"
            old_id = data.get('id', 'N/A')
            data['id'] = new_id
            renumbered_data.append(json.dumps(data, ensure_ascii=False))
            
            # นับ domain
            domain = data.get('domain', 'unknown')
            domain_count[domain] = domain_count.get(domain, 0) + 1
            
            # แสดงความคืบหน้า
            if counter % 100 == 0:
                print(f"ดำเนินการแล้ว {counter} รายการ...")
        
        except json.JSONDecodeError as e:
            print(f"⚠️  ข้าม: บรรทัดที่มีปัญหา (error: {e})")
            print(f"    เนื้อหา: {line[:100]}...")
            continue
    
    # เขียนไฟล์ใหม่
    output = output_file if output_file else input_file
    print(f"\nกำลังเขียนไฟล์: {output}")
    with open(output, 'w', encoding='utf-8') as f:
        for line in renumbered_data:
            f.write(line + '\n')
    
    print(f"\n✓ เรียง ID ใหม่สำเร็จ!")
    print(f"  - จำนวนรายการ: {len(renumbered_data)}")
    print(f"  - ไฟล์ผลลัพธ์: {output}")
    if backup and output_file is None:
        print(f"  - ไฟล์สำรอง: {backup_file}")
    
    # แสดงสถิติ domain
    print(f"\n📊 สถิติ Domain ({len(domain_count)} ประเภท):")
    sorted_domains = sorted(domain_count.items(), key=lambda x: x[1], reverse=True)
    for domain, count in sorted_domains:
        percentage = (count / len(renumbered_data)) * 100
        print(f"  - {domain:15} : {count:5} รายการ ({percentage:5.1f}%)")
    
    print(f"  - รวม: {sum(domain_count.values())} รายการ")


if __name__ == "__main__":
    # กำหนดชื่อไฟล์
    input_file = "ThaiNER.jsonl"
    
    # ตรวจสอบว่าไฟล์มีอยู่หรือไม่
    if not os.path.exists(input_file):
        print(f"❌ ไม่พบไฟล์: {input_file}")
        print(f"   กรุณาตรวจสอบว่าคุณอยู่ใน directory ที่ถูกต้อง")
        exit(1)
    
    # เรียกใช้ฟังก์ชัน
    renumber_ids(
        input_file=input_file,
        output_file=None,  # เขียนทับไฟล์เดิม (มีสำรอง)
        backup=True        # สร้างไฟล์สำรอง
    )
    
    print("\nคำแนะนำ:")
    print("- ถ้าต้องการเก็บไฟล์เดิมไว้ ให้เปลี่ยน output_file เป็นชื่อไฟล์ใหม่")
    print("- ถ้าไม่ต้องการสร้างไฟล์สำรอง ให้เปลี่ยน backup=False")
