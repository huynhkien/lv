from datetime import datetime, timedelta
import re
import random
# Ngày
def parse_date_from_query(date_string):
    """Parse ngày từ string input của user"""
    if not date_string:
        return None
        
    date_string = date_string.strip().lower()
    
    # Xử lý các từ khóa đặc biệt
    if date_string in ['hôm nay', 'today', 'hiện tại']:
        return datetime.now().date()
    elif date_string in ['hôm qua', 'yesterday',]:
        return (datetime.now() - timedelta(days=1)).date()
    elif date_string in ['hôm kia']:
        return (datetime.now() - timedelta(days=2)).date()
    
    # Parse các format ngày tháng
    current_year = datetime.now().year
    
    # Format: DD/MM/YYYY hoặc DD-MM-YYYY
    pattern1 = r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})'
    match = re.match(pattern1, date_string)
    if match:
        day, month, year = map(int, match.groups())
        try:
            return datetime(year, month, day).date()
        except ValueError:
            return None
    
    # Format: DD/MM hoặc DD-MM (năm hiện tại)
    pattern2 = r'(\d{1,2})[/-](\d{1,2})'
    match = re.match(pattern2, date_string)
    if match:
        day, month = map(int, match.groups())
        try:
            return datetime(current_year, month, day).date()
        except ValueError:
            return None
    
    # Format: chỉ có số (coi như ngày trong tháng hiện tại)
    pattern3 = r'^(\d{1,2})$'
    match = re.match(pattern3, date_string)
    if match:
        day = int(match.group(1))
        current_date = datetime.now()
        try:
            return datetime(current_date.year, current_date.month, day).date()
        except ValueError:
            return None
    
    return None

# Tháng
def parse_month_from_query(month_string):
    """Parse tháng từ string input của user"""
    if not month_string:
        return None
        
    month_string = month_string.strip().lower()
    
    # Xử lý các từ khóa đặc biệt
    if month_string in ['này', 'hiện tại', 'current', 'nay']:
        now = datetime.now()
        return now.year, now.month
    elif month_string in ['trước', 'last', 'trước đó']:
        now = datetime.now()
        if now.month == 1:
            return now.year - 1, 12
        else:
            return now.year, now.month - 1
    
    current_year = datetime.now().year
    
    # Format: MM/YYYY
    pattern1 = r'(\d{1,2})/(\d{4})'
    match = re.match(pattern1, month_string)
    if match:
        month, year = map(int, match.groups())
        if 1 <= month <= 12:
            return year, month
    
    # Format: chỉ số tháng (năm hiện tại)
    pattern2 = r'^(\d{1,2})$'
    match = re.match(pattern2, month_string)
    if match:
        month = int(match.group(1))
        if 1 <= month <= 12:
            return current_year, month
    
    return None

# Năm
def parse_year_from_query(year_string):
    """Parse năm từ string input của user"""
    if not year_string:
        return None
        
    year_string = year_string.strip().lower()
    
    # Xử lý các từ khóa đặc biệt
    if year_string in ['này', 'hiện tại', 'current', 'nay']:
        return datetime.now().year
    elif year_string in ['trước', 'last', 'ngoái', 'trước đó']:
        return datetime.now().year - 1
    elif year_string in ['kế tiếp', 'sau', 'next', 'tiếp theo']:
        return datetime.now().year + 1
    
    # Format: YYYY (4 chữ số)
    pattern1 = r'^(\d{4})$'
    match = re.match(pattern1, year_string)
    if match:
        year = int(match.group(1))
        # Kiểm tra năm hợp lệ (từ 2000 đến 2030)
        if 2000 <= year <= 2030:
            return year
    
    # Format: YY (2 chữ số cuối)
    pattern2 = r'^(\d{2})$'
    match = re.match(pattern2, year_string)
    if match:
        year_suffix = int(match.group(1))
        if year_suffix <= 30:  # 00-30 = 2000-2030
            return 2000 + year_suffix
        else:  # 31-99 = 1931-1999
            return 1900 + year_suffix
    
    return None

# Format input khi nhập vào chatbot
def format_message(text):
    spelling = {
    "chao": "chào",
    "chaof": "chào",
    "cam": "cảm",
    "on": "ơn",
    "cau": "câu",
    "cấu": "câu",
    "cầu": "câu",
    "hoi": "hỏi",
    "khong": "không",
    "hông": "không",
    "hong": "không",
    "loi": "lợi",
    "van": "vấn",
    "vân": "vấn",
    "lơi": "lợi",
    "lời": "lợi",
    "lới": "lợi",
    "lơns": "lớn",
    "lơn": "lớn",
    "lon": "lớn",
    "nhuan": "nhuận",
    "nhuân": "nhuận",
    "sap": "sắp",
    "săp": "sắp",
    "hêt": "hết",
    "het": "hết",
    "thap": "thấp",
    "thâp": "thấp",
    "thâpf": "thấp",
    "mat": "mặt",
    "mặt": "mặt",
    "đen": "đến",
    "đên": "đến",
    "dên": "đến",
    "den": "đến",
    "dến": "đến",
    "dén": "đến",
    "hang": "hàng",
    "san": "sản",
    "sàn": "sản",
    "pham": "phẩm",
    "ton": "tồn",
    "toi": "tôi",
    "danh": "doanh",
    "tui": "tôi",
    "can": "cần",
    "cân": "cần",
    "hoa": "hóa",
    "hom": "hôm",
    "hay": "hãy",
    "ten": "tên",
    "ke": "kê",
    "kế": "kê",
    "thong ke": "thống kê",
    "bao": "báo",
    "bào": "báo",
    "nhieu": "nhiều",
    "nhiều": "nhiều",
    "it": "ít",
    "du": "dư",
    "nhieu": "nhiêu",
    "thua": "thua",
    "thưa": "thừa",
    "loi nhuan": "lợi nhuận",
    "tong thu": "tổng thu",
    "phi": "phí",
    "thu nhap": "thu nhập",
    "gia tri": "giá trị",
    "gia": "giá",
    "tien": "tiền",
    "thue": "thuế",
    "don": "đơn",
    "hang": "hàng",
    "xu": "xử",
    "xư": "xử",
    "xữ": "xử",
    "ly": "lý",
    "lỳ": "lý",
    "da": "đã",
    "đả": "đã",
    "đa": "đã",
    "nhan": "nhận",
    "huy": "hủy",
    "hũy": "hủy",
    "luong": "lượng",
    "lương": "lượng",
    "luot": "lượt",
    "lươt": "lượt",
    "lướt": "lượt",
    "danh": "đánh",
    "gia": "giá",
    "sao": "sao",
    "cào": "cáo",
    "diem": "điểm",
    "nhap": "nhập",
    "xuat": "xuất",
    "sdt": "số điện thoại",
    "emai": "email",
    "emall": "email",
    "emali": "email",
    "diachi": "địa chỉ",
    "manv": "mã nhân viên",
    "tennv": "tên nhân viên",
    "chucvu": "chức vụ",
    "phongban": "phòng ban",
    "luong": "lương",
    "ngayvaolam": "ngày vào làm",
    "taikhoan": "tài khoản",
    "tai": "tài",
    "dang": "đang",
    "ky": "ký",
    "mơi": "mới",
    "moi": "mới",
    "tu": "từ",
    "tư": "từ",
    "tưf": "từ",
    "khoan": "khoản",
    "khoãn": "khoản",
    "matkhau": "mật khẩu",
    "dangnhap": "đăng nhập",
    "dangxuat": "đăng xuất",
    "dangky": "đăng ký",
    "uudai": "ưu đãi",
    "khuyenmai": "khuyến mãi",
    "giamgia": "giảm giá",
    "magiam": "mã giảm",
    "ngay": "ngày",
    "thang": "tháng",
    "nam": "năm",
    "homnay": "hôm nay",
    "homqua": "hôm qua",
    "tuannay": "tuần này",
    "thangnay": "tháng này",
    "quynam": "quý năm",
    "thoigian": "thời gian",
    "timkiem": "tìm kiếm",
    "tim": "tìm",
    "kiem": "kiếm",
    "kiêm": "kiếm",
    "loc": "lọc",
    "tiet": "tiết",
    }
    words = text.split()  
    corrected_words = []
    for word in words:
        corrected_word = spelling.get(word.lower(), word)
        corrected_words.append(corrected_word)
    return ' '.join(corrected_words)


print(format_message('tôi cần thông tin các sản phẩm tồn kho cao'))