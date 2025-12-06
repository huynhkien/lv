const Order = require('../models/order.model');
const Notification = require('../models/notification.model');
const Product = require('../models/product.model');
const User = require('../models/user.model');
const WaveHouse = require('../models/warehouse.model');
const Voucher = require('../models/voucher.model');
const asyncHandler = require('express-async-handler');

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
  console.log(status)
  // Tạo thông báo
  await Notification.create({
    type: 'order',
    status_order: {
        status: 'Processing',
        order: response._id,
        uid: _id
    }
  })
  return res.status(200).json({
        success: true,
        mes: response ? 'Đơn hàng sẽ được vận chuyển sớm nhất, Cảm ơn quý khách đã mua hàng' : 'Vui lòng thử lại'
    });
});

const updateStatus = asyncHandler(async (req, res) => {
  const { oid } = req.params;
  const {_id} = req.user;
  const { status, location } = req.body;

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
    // Tạo thông báo
    await Notification.create({
        type: 'order',
        status_order: {
            status: status,
            order: oid,
            uid: _id
        }
    })

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
}