import moment from 'moment';

function NotificationAdmin({notification, users}) {
  const filteredNotifications = notification?.filter(el => 
    el.voucher || 
    el.product || 
    el.status_order
  );

  return (
    <div className="notification">
      {filteredNotifications && filteredNotifications.length > 0 ? (
        filteredNotifications.map((el, index) => (
          <div key={index} className="notification--item">
            <p className="message">
              {el.voucher && (
                <>
                  Mã giảm giá <span className="text-primary font-weight">{el.voucher.code}</span> mới được tạo. Áp dụng từ ngày{" "}
                  {moment(el.voucher.startDate).format("DD/MM")} - {moment(el.voucher.endDate).format("DD/MM")}.{" "}
                </>
              )}
              {el.status_order  && (
                <>
                  {el.status_order?.order ? `Mã đơn hàng ${el?.status_order?.order} của khách hàng ${users.find(item => item?._id === el?.status_order?.uid)?.name}` : `Đơn hàng đang xử lý, thời gian giao dự kiến mất khoảng 2 - 3 ngày`}
                  {(() => {
                    switch(el.status_order.status) {
                      case 'Delivering':
                        return ' - đã thay đổi trạng thái "Đang được giao", vui lòng nhắc khách hàng nghe máy khi shipper gọi';
                      case 'Processing':
                        return ' - Đang xử lý, vui lòng xử lý đơn hàng';
                      case 'Cancelled':
                        return ' - đã thay đổi trạng thái "Đã hủy"';
                      case 'Succeed':
                        return ' - đã thay đổi trạng thái "Giao hàng thành công"';
                      case 'Confirm':
                        return ' - đã thay đổi trạng thái "Đã xác nhận", vui lòng nhắc khách xác nhận để nhận hàng';
                      default:
                        return '';
                    }
                  })()}
                </>
              )}
              {el.product && (
                <>
                  <a href={`http://localhost:3000/detail/${el.product._id}`} className="text-primary font-weight">{el.product.name}</a> vừa được thêm. Quý khách hàng có thể tham khảo.
                </>
              )}
            </p>
          </div>
          ))
      ) : (
        <div className="tpsection">
          <p>Không có thông báo nào hết</p>
        </div>
      )}

    </div>
  );
}

export default NotificationAdmin;
