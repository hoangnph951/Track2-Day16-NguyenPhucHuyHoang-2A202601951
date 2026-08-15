Thời gian apply: 233 giây
Kết quả benchmark cho thấy LightGBM có thời gian huấn luyện khá nhanh trên CPU, chỉ khoảng **5,67 giây** với 227.845 mẫu training và 30 đặc trưng.  
Thời gian tải toàn bộ dataset là khoảng **1,98 giây**, cho thấy dữ liệu có kích thước vừa phải và có thể xử lý tốt trên máy `e2-medium`.  
Mô hình đạt **AUC-ROC = 0,8776**, cho thấy khả năng phân biệt giữa giao dịch bình thường và giao dịch gian lận ở mức khá tốt.  
Accuracy đạt **99,86%**, tuy nhiên chỉ số này cần được xem xét thận trọng do dataset có mức mất cân bằng lớp rất cao.  
Các chỉ số **F1-Score = 0,6385**, Precision = **0,5913** và Recall = **0,6939** phản ánh rõ hơn hiệu quả phát hiện gian lận của mô hình.  
Inference latency cho một giao dịch chỉ khoảng **0,904 ms**, cho thấy mô hình có khả năng phản hồi rất nhanh trên CPU.  
Khi dự đoán theo batch 1.000 giao dịch, throughput đạt khoảng **144.345 mẫu/giây**, chứng tỏ LightGBM có hiệu năng inference rất cao ngay cả khi không sử dụng GPU.  
Nhìn chung, cấu hình CPU hiện tại đáp ứng tốt yêu cầu huấn luyện và suy luận cho bài toán ML dạng dữ liệu bảng này.