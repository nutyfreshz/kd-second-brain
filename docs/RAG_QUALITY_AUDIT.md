# KD Second Brain: audit และปรับปรุงคุณภาพ RAG

วันที่: 2026-10-05 | Baseline: `82f1fc4` | เป้าหมาย: ถามตอบจาก Markdown ที่ผ่านการ review บน Databricks Apps

## ข้อสรุป

โครงสร้าง core แยกจาก Streamlit เหมาะสำหรับต่อยอดเป็น custom webchat อยู่แล้ว จุดที่ควรลงทุนก่อนคือค้นหลักฐานให้ครบ คุมข้อความที่แสดง และตรวจย้อนกลับได้ การเพิ่ม prompt อย่างเดียวไม่แก้ปัญหาเหล่านี้

รอบนี้แก้ข้อผิดพลาดที่พิสูจน์ได้ เพิ่ม regression tests และ benchmark แต่ **ยังไม่รับรองว่าคุณภาพคำตอบเทียบเท่า NotebookLM** เพราะไม่มี provider credentials, corpus จริง หรือ Databricks runtime สำหรับทดสอบ inference ใน environment นี้

ใช้พฤติกรรมหลักจาก [Google Help](https://support.google.com/notebooklm/answer/16179559?hl=en) เป็นเกณฑ์: เลือกแหล่งข้อมูล, ตอบตามแหล่งข้อมูล, มี citation ให้ตรวจต้นฉบับ ไม่ได้ตั้งเป้าลอกทุกฟีเจอร์ เช่น Audio Overview หรือ video

## ผลตรวจและสิ่งที่แก้

| ความสำคัญ | ปัญหาที่พบใน baseline | การแก้และผลที่คาดหวัง |
|---|---|---|
| สูง | คัด top candidates ก่อนกรอง source ทำให้เอกสารที่เลือกหายจากผลค้นหา | กรองก่อน top-k ทั้ง BM25 และ semantic |
| สูง | `answer` ไม่มี claims ผ่าน validator ได้ และ `answer_th` ไม่ใช่ข้อความที่ตรวจ | บังคับ nonempty claims; แสดงเฉพาะ claims ที่ตรวจ พร้อมเลข citation |
| สูง | เช็กเพียง citation ID ไม่ตรวจตัวเลข/quote | ตรวจ quote กับ chunk จริง และตัวเลขที่มีหน่วยสำคัญต่อหลักฐานของ claim นั้น |
| สูง | free route มี paid fallback แม้ปิด paid gate | ตรวจสิทธิ์ free/paid ทุกตัวใน fallback list |
| สูง | concurrent duplicate ผ่านก่อน lock ทำให้เรียกโมเดลซ้ำ | lock ครบทั้ง turn; ทดสอบ 5 concurrent duplicates เรียก provider ครั้งเดียว |
| กลาง | Semantic encoder ตัดข้อความยาวท้าย chunk | แบ่ง token windows มี overlap แล้วรวมคะแนนกลับ parent chunk |
| กลาง | BM25 scan ทุก chunk ต่อ query และแยก P2/3BB เป็นคนละ token | inverted index อ่านเฉพาะ postings ที่เกี่ยวข้อง และเก็บ codes เต็ม |
| กลาง | บรรทัดยาวไม่ถูกแบ่ง, หัวข้อแม่หาย, ตารางขาดชื่อคอลัมน์เมื่อแตก chunk | hard bound 1,600 ตัวอักษร, heading path, เก็บ column headers ใน table fragments |
| กลาง | citation ID เดิมเปลี่ยนเนื้อหาหลัง sync และ API อ่านจาก snapshot ล่าสุด | ID มี content hash; เปิด citation ที่เก็บไว้กับคำตอบเดิมเท่านั้น |
| กลาง | คำถามใหม่สั้น ๆ ถูกผูกหัวข้อเดิมทุกครั้ง | ผูก context เมื่อพบ follow-up cues หรือมีคำถาม clarification ค้าง |
| กลาง | cache ไม่แยก effective date/context/prompt จริง, cache failures และไม่มีขนาดสูงสุด | key ครบขึ้น, จำกัด 256 entries, ไม่ cache availability/validation failures; cache hit ข้าม retrieval |
| กลาง | sync ไม่เปลี่ยนข้อมูลยังสร้าง embeddings ใหม่ | fingerprint รวม source identity และ metadata; reuse snapshot เมื่อไม่เปลี่ยน; cache model instances |
| กลาง | README กล่าวว่ามี source selection แต่ UI ค้นทั้งหมดเสมอ | เพิ่มตัวเลือกทั้งหมด/รายเอกสาร, empty selection ปิด chat input |
| กลาง | provider output ถูกตัดหรือ safety finish ไม่ถูกตรวจครบ | ปฏิเสธ truncated output; Gemini รวม text parts และไม่แสดง thought parts; key อยู่ใน header |
| กลาง | JSON metadata `"false"` กลายเป็น truthy consent | ส่ง external LLM ได้เฉพาะ boolean true หรือ explicit runtime override |

## หลักฐานการทดสอบ

- Baseline: 21 tests ผ่าน
- หลังแก้: 56 tests ผ่าน รวม Streamlit AppTest (source selection → chat → citation → new chat)
- Semantic window test ใช้ fake tokenizer/encoder เพื่อพิสูจน์ว่า tail และ source scope ไม่หาย **ไม่ใช่การวัดความแม่นยำของ E5 จริง**
- ไม่มีการเรียก external LLM และไม่มีข้อมูล AIS จริงใน fixtures/benchmark
- Python compile check ผ่าน
- GitHub workflow รัน offline tests และ synthetic benchmark เมื่อเปิด PR/เปลี่ยน main

Benchmark ทำซ้ำได้:

```bash
python -m unittest discover -s tests -v
python benchmarks/retrieval_quality.py --baseline 82f1fc4
```

ผลเต็มอยู่ที่ [retrieval_benchmark.json](retrieval_benchmark.json)

| การวัด | ก่อนแก้ | หลังแก้ |
|---|---:|---:|
| Synthetic retrieval diagnostics | 8/9 | 9/9 |
| BM25 median, 5,000 chunks / 40 sparse queries | 0.657 ms | 0.004 ms |
| BM25 p95, งานเดียวกัน | 0.793 ms | 0.006 ms |

ตัวเลขนี้วัดเฉพาะการค้นคำหายากใน fixture บนเครื่องทดสอบ ไม่รวม embedding, network, generation หรือ queue และไม่ใช้กล่าวว่าแอปตอบเร็วขึ้นตามอัตราส่วนเดียวกัน เวลาตอบจริงยังขึ้นกับโมเดลเป็นหลัก ชุด 9 คำถามเป็น diagnostic regression suite ขนาดเล็ก ไม่ใช่คะแนนความถูกต้องระดับใช้งานจริง

## ข้อจำกัดที่ยังมี และ trade-offs

1. **Citation ถูกต้องไม่ได้แปลว่าเหตุผลถูกต้อง**: validator ยังตรวจไม่ได้ครบว่าข้อความ paraphrase รองรับจริง, กลับความหมายของคำว่า "ไม่", หรือเลือก policy ผิดเรื่อง ทั้งนี้ numeric check มีขอบเขตที่ระบุในโค้ด ไม่ใช่การตรวจเลขทุกชนิด
2. Numeric validation เป็นแบบ conservative: ยอมรับรูปแบบตัวเลข/หน่วยเทียบเท่าที่รองรับ แต่ผลคำนวณหรือแปลงหน่วยใหม่ที่ไม่มีใน evidence จะไม่ผ่าน สำหรับงานคำนวณควรเพิ่ม deterministic calculation tool แยกจากโมเดล
3. Semantic search ยังต้องปรับ relevance threshold และ reranking ด้วยชุดคำถามจริง ไม่ตั้ง cutoff จากค่าที่เดา RRF score เป็นอันดับผสม ไม่ใช่ probability/confidence
4. การเปรียบเทียบข้ามหลายเอกสารหรือสรุปทั้งคลังยังใช้ top-6 evidence; อาจไม่ครอบคลุมทุกข้อยกเว้น ต้องทดสอบก่อนอ้างว่า exhaustive
5. ตารางที่แถวเดี่ยวยาวเกิน budget อาจแตกเซลล์; Markdown metadata ยังเป็น parser แบบจำกัด ไม่รองรับ YAML ทุกแบบ
6. ประวัติแชตอยู่ใน memory และหายเมื่อ restart; ไม่มี streaming, note export หรือ persistence ในรอบนี้
7. Source selection ไม่ใช่ per-user document ACL ทุก identity ใน app เห็น published corpus เดียวกัน ต้องแยก authorization หากทีมมีเอกสารที่สิทธิ์ต่างกัน
8. เมื่อเลือกค้นทั้งคลัง authority conflict ในเอกสารอื่นยัง block ได้; เลือก source ที่ต้องการจะจำกัด conflict scope แต่ยังไม่เดาว่า version ไหนถูก
9. ตาราง paid/free routing ใน `app.yaml` เป็น **paid UAT ที่เปิดไว้ก่อนงานนี้** และ Drive external-LLM override เปิดอยู่ ไม่ได้เปลี่ยน deployment policy หรืออนุญาตค่าใช้จ่ายใหม่ใน audit นี้ Free-route fix ไม่ได้เปลี่ยน deployment ให้ฟรี
10. ยังไม่ได้ยืนยัน exact provider model availability/quota, Databricks network/startup, หรือ load 10 users จริง ห้ามนำ mock tests มาแทน acceptance เหล่านี้

## Contract และการนำไปใช้

- `CONTRACT_VERSION=2`: `ChatRequest.selected_source_ids=None` หมายถึงทุก authoritative source; `()` หมายถึงไม่เลือก source ถ้า custom client ส่ง `[]` เพื่อหมายถึงทั้งหมด ต้องเปลี่ยนเป็น omitted/null
- `answer_th` สำหรับ answer/conflict ประกอบจาก checked claims พร้อม `[1]`, `[2]` ตามลำดับ citations; UI แสดงเลขตรงกันใน evidence viewer
- เปิด citation จากข้อความที่เคยตอบใน conversation นั้น ไม่อ่าน chunk ปัจจุบันแทนหลักฐานเก่า
- `not_found` ใช้ข้อความมาตรฐาน; `clarify` มีคำถามที่โมเดลเสนอ โดยไม่มี answer prose ที่ยังไม่ตรวจ
- Prompt ให้ claims เป็นคำตอบที่อ่านรู้เรื่องด้วยตัวเอง เพื่อไม่ลดคุณภาพการเรียบเรียงหลังใช้ rendering แบบนี้
- ไม่เพิ่ม dependency สำหรับ retrieval; embedding windows อาจทำให้ initial sync ใช้เวลามากขึ้น แลกกับการอ่านช่วงท้ายครบ ควรวัดจริงกับ corpus ก่อน deploy
- UI integration test ใช้ Streamlit; core + synthetic benchmark ไม่ต้องมี provider credentials

## Gate ก่อนรับรองคุณภาพจริง

ใช้คำถามจริง 30 ข้อที่ผู้รู้ตรวจ expected answer/citations ไว้ก่อน: direct lookup 10, exceptions/numeric/date 8, follow-up/clarification 5, cross-document 4, out-of-scope/conflict 3

ให้ NotebookLM และแอปตอบจากเอกสารเดียวกัน โดย reviewer ไม่เห็นชื่อระบบ วัด:

- supported claims / factual claims ทั้งหมด พร้อมแยก critical numeric/policy errors
- correct evidence recall@6 และความครบของข้อยกเว้น
- abstention ถูกกรณี: รู้ว่าเมื่อไรควรถามเพิ่ม/ตอบไม่พบ
- p50/p95 end-to-end latency, provider error/rate-limit, ค่าใช้จ่ายต่อคำถาม
- 10 distinct users และ duplicate retries ใน deployment จริง

เป้าหมายเริ่มต้นที่เสนอ ไม่ใช่ผลที่ผ่านแล้ว: zero critical policy/numeric errors ในชุด acceptance, ≥95% supported claims, ≥90% required-evidence recall และกำหนด latency target จาก quota/model จริง ถ้ายังไม่ผ่านให้แก้ retrieval/curation ก่อนเพิ่ม agent หรือโมเดลหลายชั้น

## แหล่งอ้างอิงทางเทคนิค

- [Google: source-scoped chat และ citation](https://support.google.com/notebooklm/answer/16179559?hl=en)
- [intfloat multilingual-e5-small model card](https://huggingface.co/intfloat/multilingual-e5-small): passage/query prefixes และข้อจำกัดข้อความยาว 512 tokens
- [E5 tokenizer config](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/tokenizer_config.json)
