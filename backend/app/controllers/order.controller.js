const { query, response } = require('express');
const Order = require('../models/order.model');
const Product = require('../models/product.model');
const User = require('../models/user.model');
const WaveHouse = require('../models/warehouse.model');
const Voucher = require('../models/voucher.model');
const asyncHandler = require('express-async-handler');
const config = require('../config/index');
const moment = require('moment');
const querystring = require('querystring');
const crypto = require('crypto');
const axios = require('axios');

const createOrder = asyncHandler(async (req, res) => {
  const { _id } = req.user;  
  const { products, total, status, applyVoucher } = req.body; 
  const user = await User.findById(_id);
  const voucher = await Voucher.findById(applyVoucher);

  const VoucherCount = user?.voucher?.find(v => v.voucherId.toString() === applyVoucher);
  const currentDate = new Date();
  if(currentDate > new Date(voucher?.endDate)){
    return res.status(400).json({
      success: false,
      mes: 'Voucher đã hết hạn sử dụng'
    })
  }
  if(VoucherCount && VoucherCount?.useCount > voucher?.usedCount ){
      return res.status(400).json({
        success: false,
        mes: 'Voucher đã đạt giới hạn sử dụng'
      })
  }
  if(total >= VoucherCount?.minPurchaseAmount){
    return res.status(400).json({
      success: false,
      mes: `Giá trị đơn hàng tối thiểu ${VoucherCount?.minPurchaseAmount}`
    })
  }
  // Tạo đơn hàng (order)
  const response = await Order.create({ products, total, orderBy: _id, status, applyVoucher });
  if (response) {
    const io = req.app.get('io');
    io.emit('order_status_create', {
      success: true,
      message: `Trạng thái đơn hàng đã thay đổi thành: ${status}`,
      order: {
        status: status,
        uid: _id
      }
    });
  }
  // Lặp qua từng sản phẩm trong đơn hàng để cập nhật kho
  for (const item of products) {
    const warehouse = await WaveHouse.findOneAndUpdate(
            {
                "products.product": item.product,  // Tìm sản phẩm theo ID sản phẩm
                "products.variant": item.variant, // Tìm variant của sản phẩm
                 type: 'Phiếu nhập'  ,
            },
            {
                $inc: { "products.$.quantity": -item.quantity }  // Giảm số lượng sản phẩm trong kho
            },
            { new: true } 
        );
        
    if (!warehouse) {
            throw new Error('Không tìm thấy');
        }
        const productInWarehouse = warehouse.products.find(
            (p) => p.product === item.product && p.variant === item.variant
        );

    if (productInWarehouse.quantity < 0) {
            throw new Error('Sản phẩm hết hàng');
        }
    }
    
  // Cập nhật voucher
  if (applyVoucher) {
        voucher.usedCount += 1;
        await voucher.save();
    }

  // Cập nhật điểm tích lũy của người dùng
  const totalPoints = products.reduce((acc, item) => acc + item.quantity, 0);
    
    // Lưu voucher áp dụng vào người dùng
  if (applyVoucher) {
      const userVoucher = user.voucher.find(v => v.voucherId.toString() === applyVoucher);
      if (userVoucher) {
        userVoucher.useCount += 1; 
        userVoucher.createdAt = new Date();
      } else {
        user.voucher.push({ voucherId: applyVoucher, useCount: 1, createdAt: new Date() });
      }
      await user.save();
    }

  user.accumulate.points += totalPoints;
  await user.save();

  return res.status(200).json({
        success: true,
        mes: response ? 'Đơn hàng sẽ được vận chuyển sớm nhất, Cảm ơn quý khách đã mua hàng' : 'Vui lòng thử lại'
    });
});

// Helper function để sort object
function sortObject(obj) {
    const sorted = {};
    const keys = Object.keys(obj).sort();
    keys.forEach(key => {
        sorted[key] = obj[key];
    });
    return sorted;
}

const createVnPayOrder = asyncHandler(async(req, res) => {
    // Kiểm tra timeout
    if (req.timedout || req.headers['x-timeout'] || req.headers['x-gateway-error']) {
        return res.redirect(`${process.env.URL_CLIENT}`);
    }
    
    // Validate request body
    if(!req.body || !req.body.products || !req.body.total){
        return res.status(400).json({
            success: false,
            message: 'Thông tin đơn hàng không hợp lệ'
        });
    }
    
    const { _id } = req.user;
    const { products, total, status, applyVoucher, code } = req.body;
    
    // Validate user
    const user = await User.findById(_id);
    if(!user){
        return res.status(404).json({
            success: false,
            message: 'Không tìm thấy người dùng'
        });
    }
    
    // Validate và kiểm tra voucher nếu có
    let voucher = null;
    if(applyVoucher){
        voucher = await Voucher.findById(applyVoucher);
        if(!voucher){
            return res.status(404).json({
                success: false,
                mes: 'Voucher không tồn tại'
            });
        }
        
        // Kiểm tra voucher hết hạn
        const currentDate = new Date();
        if(currentDate > new Date(voucher.endDate)){
            return res.status(400).json({
                success: false,
                mes: 'Voucher đã hết hạn sử dụng'
            });
        }
        
        // Kiểm tra giới hạn sử dụng chung của voucher
        if(voucher.usageLimit && voucher.usedCount >= voucher.usageLimit){
            return res.status(400).json({
                success: false,
                mes: 'Voucher đã đạt giới hạn sử dụng'
            });
        }
        
        // Kiểm tra số lần user đã dùng voucher này
        const userVoucherUsage = user.voucher?.find(v => v.voucherId.toString() === applyVoucher);
        if(userVoucherUsage && userVoucherUsage.useCount >= voucher.userUsageLimit){
            return res.status(400).json({
                success: false,
                mes: 'Bạn đã sử dụng hết lượt voucher này'
            });
        }
        
        // FIX: Sửa logic kiểm tra giá trị đơn hàng tối thiểu (< thay vì >=)
        if(total < voucher.minPurchaseAmount){
            return res.status(400).json({
                success: false,
                mes: `Giá trị đơn hàng tối thiểu ${voucher.minPurchaseAmount.toLocaleString('vi-VN')}đ`
            });
        }
    }
    
    // Kiểm tra tồn kho trước khi tạo đơn
    for (const item of products) {
        const warehouse = await WaveHouse.findOne({
            "products.product": item.product,
            "products.variant": item.variant,
            type: 'Phiếu nhập'
        });
        
        if (!warehouse) {
            return res.status(404).json({
                success: false,
                message: `Không tìm thấy sản phẩm ${item.name || ''} trong kho`
            });
        }
        
        const productInWarehouse = warehouse.products.find(
            (p) => p.product.toString() === item.product.toString() && p.variant === item.variant
        );
        
        if (!productInWarehouse || productInWarehouse.quantity < item.quantity) {
            return res.status(400).json({
                success: false,
                message: `Sản phẩm ${item.name || ''} không đủ hàng trong kho`
            });
        }
    }
    
    // Tạo đơn hàng trước
    const newOrder = await Order.create({
        code: code || `ORD${Date.now()}`,
        products,
        total,
        status: status || 'Processing',
        applyVoucher: applyVoucher || null,
        orderBy: _id
    });
    
    if(!newOrder){
        return res.status(500).json({
            success: false,
            message: 'Không thể tạo đơn hàng'
        });
    }
    console.log(newOrder)
    
    process.env.TZ = 'Asia/Ho_Chi_Minh';
    
    const createDate = moment(newOrder.createdAt).format('YYYYMMDDHHmmss');
    const ipAddr = req.headers['x-forwarded-for']?.split(',')[0]?.trim() || 
                   req.connection?.remoteAddress || 
                   req.socket?.remoteAddress || 
                   '127.0.0.1';
    
    const tmnCode = config.vnp_TmnCode.code;
    const secretKey = config.vnp_HashSecret.secret;
    const vnpUrl = config.vnp_Url.url;
    const returnUrl = config.vnp_Url_Return.url;
    const orderId = newOrder._id.toString();
    
    // Tạo params
    let vnp_Params = {
        vnp_Version: '2.1.0',
        vnp_Command: 'pay',
        vnp_TmnCode: tmnCode,
        vnp_Locale: 'vn',
        vnp_CurrCode: 'VND',
        vnp_TxnRef: orderId,
        vnp_OrderInfo: `Thanh toan hoa don cho GD:${orderId}`,
        vnp_OrderType: 'other',
        vnp_Amount: Math.round(newOrder.total * 100),
        vnp_ReturnUrl: returnUrl,
        vnp_IpAddr: ipAddr,
        vnp_CreateDate: createDate
    };
    
    // Sắp xếp params
    vnp_Params = sortObject(vnp_Params);
    
    // Tạo chữ ký
    const signData = Object.keys(vnp_Params)
        .map(key => `${key}=${vnp_Params[key]}`)
        .join('&');
    
    const hmac = crypto.createHmac("sha512", secretKey);
    const signed = hmac.update(Buffer.from(signData, 'utf-8')).digest("hex");
    
    // Tạo query string thủ công
    const queryParams = Object.keys(vnp_Params)
        .map(key => `${key}=${encodeURIComponent(vnp_Params[key])}`)
        .join('&');
    
    // URL cuối cùng
    const finalUrl = `${vnpUrl}?${queryParams}&vnp_SecureHash=${signed}`;
    
    console.log('VNPay URL:', finalUrl);
    
    if (!finalUrl) {
        throw new Error('Tạo URL thanh toán không thành công');
    }
    
    // Cập nhật kho và voucher sau khi tạo URL thanh toán thành công
    try {
        // Cập nhật số lượng trong kho
        for (const item of products) {
            await WaveHouse.findOneAndUpdate(
                {
                    "products.product": item.product,
                    "products.variant": item.variant,
                    type: 'Phiếu nhập',
                },
                {
                    $inc: { "products.$.quantity": -item.quantity }
                },
                { new: true } 
            );
        }
        
        // Cập nhật voucher nếu có
        if (applyVoucher && voucher) {
            voucher.usedCount += 1;
            await voucher.save();
            
            // Cập nhật voucher của user
            const userVoucherIndex = user.voucher?.findIndex(v => v.voucherId.toString() === applyVoucher);
            if (userVoucherIndex !== -1) {
                user.voucher[userVoucherIndex].useCount += 1;
                user.voucher[userVoucherIndex].createdAt = new Date();
            } else {
                if(!user.voucher) user.voucher = [];
                user.voucher.push({ 
                    voucherId: applyVoucher, 
                    useCount: 1, 
                    createdAt: new Date() 
                });
            }
        }
        
        // Cập nhật điểm tích lũy
        const totalPoints = products.reduce((acc, item) => acc + item.quantity, 0);
        if(!user.accumulate) user.accumulate = { points: 0 };
        user.accumulate.points += totalPoints;
        await user.save();
        
        // Emit socket event
        const io = req.app.get('io');
        if(io){
            io.emit('order_status_create', {
                success: true,
                message: `Đơn hàng mới được tạo`,
                order: {
                    _id: newOrder._id,
                    status: newOrder.status,
                    uid: _id
                }
            });
        }
        
        return res.status(200).json({
            success: true,
            message: 'Tạo đơn hàng thành công',
            orderId: orderId,
            paymentUrl: finalUrl
        });
        
    } catch (error) {
        console.error('Create VNPay order error:', error);
        // Xóa đơn hàng nếu có lỗi
        await Order.findByIdAndDelete(newOrder._id);
        return res.status(500).json({
            success: false,
            message: 'Lỗi khi xử lý đơn hàng: ' + error.message
        });
    }
});

const vnpRefund = asyncHandler(async(req, res) => {
    const { oid } = req.params;
    const { _id } = req.user;
    
    const order = await Order.findById(oid).populate('products.product');
    if(!order) {
        return res.status(404).json({
            success: false,
            message: 'Không tìm thấy thông tin đơn hàng'
        });
    }
    
    process.env.TZ = 'Asia/Ho_Chi_Minh';
    let date = new Date();

    let vnp_TmnCode = config.vnp_TmnCode.code;
    let secretKey = config.vnp_HashSecret.secret;
    let vnp_Api = config.vnp_Api.url;
    
    let vnp_TxnRef = oid;
    let vnp_TransactionDate = moment(order.createdAt).format('YYYYMMDDHHmmss');
    let vnp_Amount = (req.body.amount || order.total) * 100;
    let vnp_TransactionType = req.body.transType || '02'; // 02 = hoàn trả toàn phần
    let vnp_CreateBy = req.body.user || _id.toString();
            
    let currCode = 'VND';
    let vnp_RequestId = moment(date).format('HHmmss');
    let vnp_Version = '2.1.0';
    let vnp_Command = 'refund';
    let vnp_OrderInfo = 'Hoan tien GD ma:' + vnp_TxnRef;
            
    let vnp_IpAddr = req.headers['x-forwarded-for']?.split(',').shift() 
                     || req.connection?.remoteAddress 
                     || req.socket?.remoteAddress 
                     || '127.0.0.1';
    let vnp_CreateDate = moment(date).format('YYYYMMDDHHmmss');
    let vnp_TransactionNo = '0';
    
    let data = vnp_RequestId + "|" + vnp_Version + "|" + vnp_Command + "|" + vnp_TmnCode + "|" + vnp_TransactionType + "|" + vnp_TxnRef + "|" + vnp_Amount + "|" + vnp_TransactionNo + "|" + vnp_TransactionDate + "|" + vnp_CreateBy + "|" + vnp_CreateDate + "|" + vnp_IpAddr + "|" + vnp_OrderInfo;
    let hmac = crypto.createHmac("sha512", secretKey);
    let vnp_SecureHash = hmac.update(Buffer.from(data, 'utf-8')).digest("hex");
    
    let dataObj = {
        'vnp_RequestId': vnp_RequestId,
        'vnp_Version': vnp_Version,
        'vnp_Command': vnp_Command,
        'vnp_TmnCode': vnp_TmnCode,
        'vnp_TransactionType': vnp_TransactionType,
        'vnp_TxnRef': vnp_TxnRef,
        'vnp_Amount': vnp_Amount,
        'vnp_TransactionNo': vnp_TransactionNo,
        'vnp_CreateBy': vnp_CreateBy,
        'vnp_OrderInfo': vnp_OrderInfo,
        'vnp_TransactionDate': vnp_TransactionDate,
        'vnp_CreateDate': vnp_CreateDate,
        'vnp_IpAddr': vnp_IpAddr,
        'vnp_SecureHash': vnp_SecureHash,
    };
    
    try {
        const vnpayResponse = await axios.post(vnp_Api, dataObj, {
            headers: {
                'Content-Type': 'application/json'
            },
            timeout: 30000 
        });
        
        // Kiểm tra response từ VNPay
        if (vnpayResponse.data.vnp_ResponseCode !== '00') {
            return res.status(400).json({
                success: false,
                message: 'VNPay hoàn tiền thất bại',
                error: vnpayResponse.data
            });
        }
        
        // Cập nhật trạng thái đơn hàng thành Cancelled
        const { status, location } = req.body;
        const updatedOrder = await Order.findByIdAndUpdate(
            oid, 
            { 
                status: 'Cancelled', 
                location: location || order.location 
            }, 
            { new: true }
        );
        
        if (updatedOrder) {
            const io = req.app.get('io');
            if(io){
                io.emit('order_status_update', {
                    success: true,
                    message: `Đơn hàng ${oid} đã được hoàn tiền và hủy`,
                    order: {
                        status: 'Cancelled',
                        oid: oid,
                        uid: updatedOrder.orderBy
                    }
                });
            }
        }

        // Khôi phục số lượng sản phẩm trong kho
        for (const item of order.products) {
            await WaveHouse.findOneAndUpdate(
                {
                    "products.product": item.product._id,
                    "products.variant": item.variant,
                    type: 'Phiếu nhập'
                },
                {
                    $inc: { "products.$.quantity": item.quantity }
                },
                { new: true }
            );
        }
        
        // Trừ điểm tích lũy của người dùng
        const totalPoints = order.products.reduce((acc, item) => acc + item.quantity, 0);
        const user = await User.findById(order.orderBy);
        if(user && user.accumulate){
            user.accumulate.points = Math.max(0, user.accumulate.points - totalPoints);
            await user.save();
        }

        return res.status(200).json({
            success: true,
            message: 'Hoàn tiền thành công',
            vnpayData: vnpayResponse.data
        });
        
    } catch (error) {
        console.error('VNPay Refund Error:', error.message);
        return res.status(500).json({
            success: false,
            message: 'Lỗi khi gọi API VNPay',
            error: error.message
        });
    }
});

// Trả về kết quả thanh toán với vnpay
const vnReturn = asyncHandler(async(req, res) => {
    let vnp_Params = { ...req.query };
    
    let secureHash = vnp_Params['vnp_SecureHash'];
    let orderId = vnp_Params['vnp_TxnRef'];
    let responseCode = vnp_Params['vnp_ResponseCode'];
    
    // Validate params
    if (!vnp_Params || Object.keys(vnp_Params).length === 0 || !secureHash) {
        return res.redirect(`${process.env.URL_CLIENT}/checkout/return-order-vnp/97`);
    }
    
    // Người dùng hủy thanh toán
    if (responseCode === '24') {
        // Xóa đơn hàng nếu user hủy thanh toán
        if(orderId){
            await Order.findByIdAndDelete(orderId);
        }
        return res.redirect(`${process.env.URL_CLIENT}/checkout/return-order-vnp/24`);
    }
    
    delete vnp_Params['vnp_SecureHash'];
    delete vnp_Params['vnp_SecureHashType'];
    
    vnp_Params = sortObject(vnp_Params);
    
    let tmnCode = config.vnp_TmnCode.code;
    let secretKey = config.vnp_HashSecret.secret;
    
    let signData = querystring.stringify(vnp_Params, { encode: false });
    let hmac = crypto.createHmac("sha512", secretKey);
    let signed = hmac.update(Buffer.from(signData, 'utf-8')).digest("hex");
    
    // Xác thực chữ ký
    if(secureHash === signed){
        if(responseCode === '00') {
            // Thanh toán thành công
            return res.redirect(`${process.env.URL_CLIENT}/checkout/return-order-vnp/00`);
        } else {
            // Thanh toán thất bại
            if(orderId){
                await Order.findByIdAndDelete(orderId);
            }
            return res.redirect(`${process.env.URL_CLIENT}/checkout/return-order-vnp/${responseCode}`);
        }
    } else {
        // Chữ ký không hợp lệ
        return res.redirect(`${process.env.URL_CLIENT}/checkout/return-order-vnp/97`);
    }
});
const updateStatus = asyncHandler(async (req, res) => {
  const { oid } = req.params;
  const {_id} = req.user;
  const { status, location } = req.body;
  console.log(oid, status)

  // Tìm đơn hàng dựa vào ID
  const order = await Order.findById(oid).populate('products.product');
  if (!order) {
    return res.status(404).json({ message: 'Đơn hàng không tồn tại' });
  }

  // Kiểm tra trạng thái đơn hàng cũ
  const previousStatus = order?.status;

  // Cập nhật trạng thái đơn hàng
  const response = await Order.findByIdAndUpdate(oid, { status, location }, { new: true });
  if (response) {
    const io = req.app.get('io');
    io.emit('order_status_update', {
      success: true,
      message: `Trạng thái đơn hàng ${oid} đã thay đổi thành: ${status}`,
      order: {
        status: status,
        oid: oid,
        uid: response?.orderBy
      }
    });
  }

  // Nếu đơn hàng chuyển trạng thái sang "Cancelled" hoặc "Returned", khôi phục lại số lượng sản phẩm trong kho
  if ((status === 'Cancelled' ) && (previousStatus !== 'Cancelled' )) {
    for (const item of order.products) {
      // Tìm sản phẩm trong kho
      const warehouse = await WaveHouse.findOneAndUpdate(
        {
          "products.product": item.product._id,  // Tìm sản phẩm theo ID sản phẩm
          "products.variant": item.variant,       // Tìm variant của sản phẩm
          type: 'Phiếu nhập'
        },
        {
          $inc: { "products.$.quantity": item.quantity }  // Tăng lại số lượng sản phẩm trong kho
        },
        { new: true }
      );

      if (!warehouse) {
        return res.status(404).json({ message: 'Không tìm thấy sản phẩm trong kho' });
      }
    }
  }
  if ((status === 'Delivering' || status === 'Succeed' || status === 'Confirm') && previousStatus !== status) {
    for (const item of order.products) {
      await Product.findByIdAndUpdate(item.product._id, {
        $inc: { sold: item.quantity }
      });
    }
  }
    // Cập nhật điểm tích lũy của người dùng
    const totalPoints = order.products.reduce((acc, item) => acc + item.quantity, 0);
    const user = await User.findById(_id); 
    user.accumulate.points -= totalPoints; 
    await user.save();

  return res.status(200).json({
    success: response ? true : false,
    message: response ? 'Cập nhật thành công' : 'Gặp vấn đề trong quá trình cập nhật',
  });
});

const getOrderUser = asyncHandler(async(req, res) => {
    const { _id } = req.user;
    const response = await Order.find({orderBy: _id}).sort({ createdAt: -1 });;
    return res.status(200).json({
        success: response ? true : false,
        message: response ? response : 'Không tìm thấy thông tin đơn hàng'
    })
})
const getOrderAdmin = asyncHandler(async (req, res) => {
  const queries = {...req.query};
  // Tách các trường đặc biệt ra khỏi query
    const excludeFields = ['limit', 'sort', 'page', 'fields'];
    excludeFields.forEach(el => delete queries[el])

    // Định dạng lại các operatirs cho đúng cú pháp của moogose
    let queryString = JSON.stringify(queries);
    queryString = queryString.replace(/\b(gte|gt|lt|lte)\b/g, matchedEl => `$${matchedEl}`);
    const  formatQueries = JSON.parse(queryString);

    // Filtering 
    if(queries?.q) {
        delete formatQueries.q;
        formatQueries.status = {$regex: queries.q, $options: 'i'}
    }
    let queryCommand = Order.find(formatQueries);

    //sorting
    if(req.query.sort) {
        const sortBy = req.query.sort.split(',').join(' ');
        queryCommand = queryCommand.sort(sortBy);
    }
    // Field Limiting
    if(req.query.fields){
        const fields = req.query.fields.split(',').join(' ');
        queryCommand = queryCommand.select(fields);
    }
    //
    const page = req.query.page * 1 || 1;
    const limit = req.query.limit * 1 || 100;
    const skip = (page - 1) * limit;
    queryCommand = queryCommand.skip(skip).limit(limit);
 
    // Execute  \query
    const queryExecute = await queryCommand.exec();
    const counts = await Order.countDocuments(formatQueries);
    return res.status(200).json({
        success: queryExecute.length > 0,
        data: queryExecute,
        counts
    });
});

const getOrderId = asyncHandler(async(req, res) => {
    const { oid } = req.params;
    const response = await Order.findById(oid);
    return res.status(200).json({
        success: response ? true : false,
        message: response ? response : 'Cannot get Order'
    })
})
// get status
const getCountStatus = asyncHandler(async (req, res) => {
    const { month, year } = req.query;
    
    // Tạo filter theo tháng và năm nếu có
    let dateFilter = {};
    if (month && year) {
        const startDate = new Date(year, month - 1, 1);
        const endDate = new Date(year, month, 0, 23, 59, 59);
        dateFilter = { createdAt: { $gte: startDate, $lte: endDate } };
    } else if (year) {
        const startDate = new Date(year, 0, 1);
        const endDate = new Date(year, 11, 31, 23, 59, 59);
        dateFilter = { createdAt: { $gte: startDate, $lte: endDate } };
    }

    const processingCount = await Order.countDocuments({ 
        status: 'Processing',
        ...dateFilter
    });
    const deliveringCount = await Order.countDocuments({ 
        status: 'Delivering',
        ...dateFilter
    });
    const cancelledCount = await Order.countDocuments({ 
        status: 'Cancelled',
        ...dateFilter
    });
    const succeedCount = await Order.countDocuments({ 
        status: 'Succeed',
        ...dateFilter
    });

    const result = {
        "Đang xử lý": processingCount,
        "Đang giao hàng": deliveringCount,
        "Đã hủy": cancelledCount,
        "Thành công": succeedCount,
    };

    res.status(200).json({
        success: true,
        data: result
    });
});

const getTotalAmountByDay = asyncHandler(async (req, res) => {
    const { day, month, year } = req.query;
    
    // Tạo match filter theo ngày, tháng, năm
    let matchFilter = {
        createdAt: { $exists: true }
    };
    
    if (year) {
        const startDate = new Date(year, 0, 1);
        const endDate = new Date(year, 11, 31, 23, 59, 59);
        
        if (month) {
            startDate.setMonth(month - 1);
            startDate.setDate(1);
            endDate.setMonth(month - 1);
            endDate.setDate(new Date(year, month, 0).getDate());
            endDate.setHours(23, 59, 59);
            
            if (day) {
                startDate.setDate(day);
                startDate.setHours(0, 0, 0);
                endDate.setDate(day);
                endDate.setHours(23, 59, 59);
            }
        }
        
        matchFilter.createdAt = { $gte: startDate, $lte: endDate };
    }
    
    const totalsByDay = await Order.aggregate([
        {
            $match: matchFilter
        },
        {
            $group: {
                _id: {
                    year: { $year: '$createdAt' },
                    month: { $month: '$createdAt' },
                    day: { $dayOfMonth: '$createdAt' }
                },
                totalAmount: { $sum: '$total' }
            }
        },
        {
            $project: {
                _id: 0,
                date: {
                    $dateToString: {
                        format: '%d/%m/%Y',
                        date: {
                            $dateFromParts: {
                                year: '$_id.year',
                                month: '$_id.month',
                                day: '$_id.day'
                            }
                        }
                    }
                },
                day: '$_id.day',
                month: '$_id.month',
                year: '$_id.year',
                totalAmount: { $multiply: ['$totalAmount', 23500] }
            }
        },
        {
            $sort: { year: 1, month: 1, day: 1 }
        }
    ]);
    
    return res.status(200).json({
        success: true,
        data: totalsByDay
    });
});
  const getTotalAmountByMonth = asyncHandler(async (req, res) => {
    const totalsByMonth = await Order.aggregate([
      {
        $match: {
          createdAt: { $exists: true } // Đảm bảo có trường createdAt trong document
        }
      },
      {
        $group: {
          _id: {
            month: { $month: '$createdAt' }, // Nhóm theo tháng trong năm của trường createdAt
            year: { $year: '$createdAt' } // Lấy năm của trường createdAt
          },
          totalAmount: { $sum: '$total' } // Tính tổng của trường total
        }
      },
      {
        $project: {
          _id: 0,
          month: '$_id.month',
          totalAmount: { $multiply: ['$totalAmount', 23500] } // Nhân tổng totalAmount với 23.500
        }
      },
      {
        $sort: { year: 1, month: 1 } // Sắp xếp theo năm và tháng
      }
    ]);
  
    return res.status(200).json({
      success: true,
      data: totalsByMonth
    });
  });
  const getTotalAmountByYear = asyncHandler(async (req, res) => {
    const totalsByYear = await Order.aggregate([
      {
        $match: {
          createdAt: { $exists: true } // Đảm bảo có trường createdAt trong document
        }
      },
      {
        $group: {
          _id: {
            year: { $year: '$createdAt' } // Lấy năm của trường createdAt
          },
          totalAmount: { $sum: '$total' } // Tính tổng của trường total
        }
      },
      {
        $project: {
          _id: 0,
          year: '$_id.year',
          totalAmount: { $multiply: ['$totalAmount', 23500] } // Nhân tổng totalAmount với 23.500
        }
      },
      {
        $sort: { year: 1 } // Sắp xếp theo năm 
      }
    ]);
  
    return res.status(200).json({
      success: true,
      data: totalsByYear
    });
  });
 const getCountOrder = asyncHandler(async(req, res) => {
    const order = await Order.countDocuments();
    return res.status(200).json({
      success: order ? true : false,
      data: order
    })

  });


module.exports = {
    createOrder,
    updateStatus,
    getOrderUser,
    getOrderAdmin,
    getOrderId,
    getCountStatus,
    getTotalAmountByDay,
    getTotalAmountByMonth,
    getTotalAmountByYear,
    getCountOrder,
    createVnPayOrder,
    vnReturn,
    vnpRefund
}