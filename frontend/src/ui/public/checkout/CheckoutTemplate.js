import { Breadcrumb, Paypal, Congratulation } from "../../../components/Index";
import withBaseComponents from "../../../hocs/withBaseComponents";
import { getCurrent } from "../../../store/user/asyncActions";
import { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import UserDetail from "../user/UserDetail";
import { apiGetVoucher, apiGetVouchers } from "../../../apis/voucher";
import VoucherItem from "../voucher/VoucherItem";
import { apiCreateOrderVnpay, apiGetAllOrder } from "../../../apis";
import Swal from "sweetalert2";
import { useCallback } from "react";
const CheckOut = ({dispatch, navigate}) => {
  const calculateTotal = (cart) => {
    return cart.reduce((sum, el) => sum + el?.price * el?.quantity, 0);
  };
  const { currentCart, current } = useSelector((state) => state.user);
  const [vouchers, setVouchers] = useState(null);
  const [voucher, setVoucher] = useState(null);
  const [orders, setOrders] = useState(null);
  const [applyVoucher, setApplyVoucher] = useState(null);
  const [selectedVoucherId, setSelectedVoucherId] = useState(null);
  const total = applyVoucher / 23.500;
  const round = Math.round(total);
  console.log(round);
  const  [isSuccess, setIsSuccess] = useState(false);
  useEffect(() => {
    if(isSuccess) dispatch(getCurrent())
  },[isSuccess])


    const fetchVouchers = async () => {
      const response = await apiGetVouchers();
      if (response.success) setVouchers(response?.data);
    }
  
    useEffect(() => {
      fetchVouchers();
    }, []); 
    
    useEffect(() => {
      const fetchVoucher = async () => {
          if (selectedVoucherId) {
              const response = await apiGetVoucher(selectedVoucherId);
              if (response.success) {
                  const fetchedVoucher = response.data;
                  setVoucher(fetchedVoucher);
                  if (calculateTotal(currentCart) >= fetchedVoucher.minPurchaseAmount) {
                      const voucherUser = fetchedVoucher?.applicableUsers?.filter(item => {
                          return item.name === current?.accumulate?.rank;
                      });
                      const voucherProduct = currentCart.filter(item => {
                                        return fetchedVoucher?.applicableProducts?.some(el => 
                                            el.product === item.product 
                                        );
                                    });
                      const voucherCategory = currentCart?.filter(item => {
                                    return fetchedVoucher?.applicableCategories?.some(el => 
                                      el.name === item.category 
                                    );
                                  });
                      const voucherAll = fetchedVoucher?.applyType === 'all';
                      
                      if (voucherUser?.length > 0) {
                          console.log(voucherUser);
                          if(fetchedVoucher?.discountType === 'percentage'){
                            const totalPriceAfterDiscount = calculateTotal(currentCart) * (fetchedVoucher.discountValue / 100);
                            setApplyVoucher(totalPriceAfterDiscount);
                          }else{
                            const totalPriceAfterDiscount = calculateTotal(currentCart) - (fetchedVoucher.discountValue);
                            setApplyVoucher(totalPriceAfterDiscount);
                          }                        
                      }else if(voucherProduct?.length > 0){
                        const discountAmount = voucherProduct.reduce((total, item) => {
                            const discount = item.price * (response?.data?.discountValue / 100);  
                            return total + (discount * item.quantity);  
                        }, 0); 
                        console.log("Tổng giảm giá:", discountAmount);
                        
                        // Tính tổng tiền mới cho cart sau khi áp dụng mã giảm giá
                        const totalPriceAfterDiscount = currentCart.reduce((total, item) => {
                            const isInVoucher = response?.data?.applicableProducts?.some(el => 
                                el.product === item.product 
                            );
                              if (isInVoucher) {
                                // Tính giảm giá chỉ một lần cho sản phẩm
                              if (fetchedVoucher?.discountType === 'percentage') {
                                const discount = item.price * (response?.data?.discountValue / 100);
                                // Áp dụng giảm giá một lần và cộng với (số lượng - 1) lần giá gốc
                                return total + (item.price - discount) + (item.price * (item.quantity - 1));
                              } else {
                                // Áp dụng giảm giá cố định một lần và cộng với (số lượng - 1) lần giá gốc
                                return total + (item.price - response?.data?.discountValue) + (item.price * (item.quantity - 1));
                              }
                            }
                      
                             return total + (item.price * item.quantity);
                            }, 0);
                           console.log("Tổng tiền sau khi giảm giá:", totalPriceAfterDiscount);
                           setApplyVoucher(totalPriceAfterDiscount);
                      }else if(voucherCategory?.length > 0){
                        // Tính tổng tiền mới cho cart sau khi áp dụng mã giảm giá
                        const totalPriceAfterDiscount = currentCart.reduce((total, item) => {
                          const isInVoucher = response?.data?.applicableCategories?.some(el => 
                            el.name === item.category 
                          );
                          console.log(isInVoucher);
                          if (isInVoucher) {
                            // Tính giảm giá chỉ một lần cho sản phẩm
                            if (fetchedVoucher?.discountType === 'percentage') {
                              const discount = item.price * (response?.data?.discountValue / 100);
                              // Áp dụng giảm giá một lần và cộng với (số lượng - 1) lần giá gốc
                              return total + (item.price - discount) + (item.price * (item.quantity - 1));
                            } else {
                              // Áp dụng giảm giá cố định một lần và cộng với (số lượng - 1) lần giá gốc
                              return total + (item.price - response?.data?.discountValue) + (item.price * (item.quantity - 1));
                            }
                          }
                      
                            return total + (item.price * item.quantity);
                        }, 0);
                    
                        console.log("Tổng tiền sau khi giảm giá:", totalPriceAfterDiscount);
                        setApplyVoucher(totalPriceAfterDiscount);
                      }else if(voucherAll){
                        if(fetchedVoucher?.discountType === 'percentage'){
                          const totalPriceAfterDiscount = calculateTotal(currentCart) * (fetchedVoucher.discountValue / 100);
                          setApplyVoucher(totalPriceAfterDiscount);
                        }else{
                          const totalPriceAfterDiscount = calculateTotal(currentCart) - (fetchedVoucher.discountValue);
                          setApplyVoucher(totalPriceAfterDiscount);
                        }      
                      }else{
                        setApplyVoucher(calculateTotal(currentCart));
                      }
                  } else {
                      console.log(`Tổng giá trị giỏ hàng phải lớn hơn hoặc bằng ${fetchedVoucher.minPurchaseAmount} để áp dụng voucher.`);
                      setApplyVoucher(calculateTotal(currentCart));
                  }
              }
          }else{
            setApplyVoucher(calculateTotal(currentCart));
          }          
      };
  
      fetchVoucher();
  }, [selectedVoucherId, currentCart]);
    
  const handleChangeVoucher = (event) => {
    const Id = event.target.value; 
  
    console.log('Selected Voucher ID:', selectedVoucherId);
    if (selectedVoucherId !== Id) {
      setSelectedVoucherId(Id); 
    }else{
      setSelectedVoucherId(null);
    }
  
  };  
  // Tao ma code tu dong
  // Hiển thị thông tin đơn hàng 
  useEffect(() => {
    const fetchOrders = async() => {
      const response = await apiGetAllOrder();
      if(response.success) setOrders(response.data || []);
    }
    fetchOrders();
  },[]);

  // Tạo mã code tự động
  const generateUniqueCode = useCallback(() => {
    const currentYear = new Date().getFullYear();
    const yearSuffix = currentYear.toString().slice(-2); // Lấy 2 số cuối của năm
    
    // Tìm số thứ tự cao nhất từ các đơn hàng hiện có
    const currentYearOrders = orders?.filter(order => 
      order.code && order.code.startsWith(`SEAFOOD${yearSuffix}-`)
    );
    if(currentYearOrders) return;
    
    let maxNumber = 0;
    currentYearOrders?.forEach(order => {
      const match = order.code.match(/SEAFOOD\d{2}-0*(\d+)$/);
      if (match) {
        const number = parseInt(match[1]);
        if (number > maxNumber) {
          maxNumber = number;
        }
      }
    });
    // Tạo mã mới với số thứ tự tiếp theo
    const nextNumber = maxNumber + 1;
    const paddedNumber = nextNumber.toString().padStart(3, '0'); 
    return `SEAFOOD${yearSuffix}-${paddedNumber}`;
  }, [orders]);
  // Thanh toan voi vnpay
   const handleCreateOrderVnpay = async () => {
    try {
      const payload = {
        code: generateUniqueCode(),
        products: currentCart,
        total: total,
        orderBy: current?._id,
        applyVoucher: selectedVoucherId
      };
      
      const response = await apiCreateOrderVnpay(payload);
      
      if (response?.success) {
        // Sửa lại: Swal.fire thay vì Swal.success
        Swal.fire({
            text: `Đặt hàng thành công!. Chuyển hướng đến VNPay...`,
            icon: 'success',
            confirmButtonText: 'Trở về',
            timer: 3000,
            timerProgressBar: true
          }).then(() => {
            window.location.href = response.paymentUrl;
          });
      } else {
        // Xử lý trường hợp thất bại
        Swal.fire({
          icon: 'error',
          title: 'Lỗi',
          text: response?.message || 'Có lỗi xảy ra khi tạo đơn hàng'
        });
      }
    } catch (error) {
      // Xử lý lỗi
      console.error('Error creating VNPay order:', error);
      Swal.fire({
        icon: 'error',
        title: 'Lỗi',
        text: 'Không thể tạo đơn hàng. Vui lòng thử lại sau.'
      });
    }
  };
   return (
    <div>
      {isSuccess && <Congratulation/>}
      <Breadcrumb category='Thanh toán' />
      <section className="checkout-area pb-50">
        <div className="container">
            <div className="row">
              <div className="col-lg-6 col-md-12">
                <div className="checkbox-form your-order mb-30">
                  <h3 className="mt-2">Thông tin</h3>
                  <div className="row">
                    <UserDetail/>
                  </div>
                </div>
                {vouchers?.length > 0 &&
                <div className="checkbox-form your-order mt-5">
                  <h3 className="mt-2">Ưu đãi</h3>
                  <div className="row">
                  <div className='select-voucher'>
                    {vouchers?.map((item) => (
                      <VoucherItem
                          voucherData={{
                            id: item?._id,
                            code: item?.code,
                            description: item?.description,
                            discountType: item?.discountType,
                            discountValue: item?.discountValue,
                            minPurchaseAmount: item?.minPurchaseAmount,
                            maxDiscount: item?.maxDiscount,
                            startDate: item?.startDate,
                            endDate: item?.endDate,
                            usageLimit: item?.usageLimit,
                            usedCount: item?.usedCount,
                            userUsageLimit: item?.userUsageLimit,
                            applyType: item?.applyType,
                            applicableProducts: item?.applicableProducts,
                            applicableCategories: item?.applicableCategories,
                            applicableUsers: item?.applicableUsers
                          }}
                          handleChange={handleChangeVoucher}
                          isSelected={selectedVoucherId === item._id}
                          id={selectedVoucherId}
                      />
                    ))}
                    </div>
                  </div>
                </div>
                }
              </div>
              <div className="col-lg-6 col-md-12">
                <div className="your-order mb-30">
                  <h3>Đơn hàng</h3>
                  <div className="your-order-table table-responsive">
                    <table>
                      <thead>
                        <tr>
                          <th className="product-name">Sản phẩm</th>
                          <th>Chọn</th>
                          <th>Số lượng</th>
                          <th>Giá</th>
                          <th className="product-total">Tổng</th>
                        </tr>
                      </thead>
                      <tbody>
                        {currentCart?.map((el) => (
                        <tr className="cart_item" key={el?._id || el?.name + el?.variant}>
                          <td className="product-name">
                           {el?.name}
                          </td>
                          <td className="product-total">
                            <span className="amount">{el?.variant}</span>
                          </td>
                          <td className="product-total">
                            <span className="amount">{el?.quantity}</span>
                          </td>
                          <td className="product-total">
                            <span className="amount">{(el?.price).toLocaleString()}</span>
                          </td>
                          <td className="product-total">
                            <span className="amount">{(el?.price * el?.quantity).toLocaleString()}</span>
                          </td>
                          
                        </tr>
                        ))}
                      </tbody>
                      <tfoot>
                        <tr className="cart-subtotal">
                          <th>Total</th>
                          <td><span className="amount">{calculateTotal(currentCart).toLocaleString()} VNĐ</span></td>
                        </tr>
                        <tr className="order-total">
                          <th>Giảm</th>
                          <td><strong><span className="amount">{((calculateTotal(currentCart) - applyVoucher)?.toLocaleString())} VNĐ</span></strong></td>
                        </tr>
                        <tr className="order-total">
                          <th>Tổng thanh toán</th>
                          <td><strong><span className="amount">{selectedVoucherId ? applyVoucher?.toLocaleString() : (calculateTotal(currentCart)?.toLocaleString())} VNĐ</span></strong></td>
                        </tr>
                        <tr className="order-total">
                          <th>Quy đổi</th>
                          <td><strong><span className="amount">{total?.toLocaleString()} $</span></strong></td>
                        </tr>
                      </tfoot>
                    </table>
                  </div>
                  <div className="payment-method">
                    <div className="accordion" id="checkoutAccordion">
                      <div>
                          <button className="btn btn-outline-primary w-100 py-3" onClick={handleCreateOrderVnpay}>
                            <span>VNPAY</span>
                          </button>
                      </div>
                      <div className="py-3 text-center"><span>Hoặc</span></div>
                      <div className="accordion-item">
                        <Paypal amount={round}
                                payload={{code: generateUniqueCode(), products: currentCart, total: total, orderBy: current?._id, applyVoucher: selectedVoucherId }}
                                setIsSuccess={setIsSuccess}
                        />
                      </div>
                    </div>
                 
                  </div>
                </div>
              </div>
            </div>
        </div>
      </section>
    </div>
  );
};

export default withBaseComponents(CheckOut);
