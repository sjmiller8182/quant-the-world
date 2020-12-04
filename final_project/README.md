
# Optimizing for a Model for Minimal Financial Loss

## Summary

To be competitive within our industry, gaining rapid insights from generating useful models is
not only profitable, but required for long-term success. Therefore, we have decided to implement
machine learning into our operations to provide solutions at which we otherwise would be unable
to arrive. The problem our business has been facing is if we should invest additional labor to
resolve data we were previously unable to deliver to our clients due to quality. Our customers
have agreed to purchase the data if it is corrected and delivered as needed - based on market
conditions - but it must be delivered at the time it is needed or we will fail to obtain full revenue.
Therefore, predicting when the client will need the data is paramount to our success.


It is important to note that if we do not act, all potential revenue from this data will remain lost.
If we correct the data, we can still recover some revenue. However, if we attempt to deliver the
data too early, we will lose 25 dollars on each dollar invested in correcting our data as we will
need to house the additional inventory until it is useful. Conversely, if we deliver the data too
late, we will lose 125 dollars because the client will no longer fully benefit from the corrected data.
In other words, predicting the data is needed when it is not yet needed (a false positive) will cost
us 25 dollars while predicting the data is not yet needed when it is (a false negative) will cost us
125 dollars.


When applying machine learning, there are many different models that can be used. Typically,
however, models perform differently than other models when applied to different datasets. Con-
sequently, we decided to solve our problem using three different candidate models, testing each
approach with multiple configurations, to find the best solution. All modeling approaches take
into consideration the fact that an early delivery costs 25 dollars and a late delivery costs 125.

[See Full Report](./Final%Project.pdf)

**Language**: python

## Data

An unlabeled dataset consisting of 160,000 records, 50 features, and a binary target variable was used for this analysis. The objective is to minimize monetary loss due to error, where false positive values represent a $25.00 cost and false negatives incur a $125.00 loss.


