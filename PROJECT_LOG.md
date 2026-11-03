# StockMind: project log

## Block 1: setup and data loading
- Status: done
- Data source: Chen, D. (2012). Online Retail II. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
- Rows loaded: 1,067,371 (8 columns)
- Date range: 2009-12-01 to 2011-12-09
- Missing values: customer_id 22.8%, description 0.4%, all other columns 0%
- Raw data issues found: 19,494 cancelled invoices, 22,950 negative-quantity rows, 6,207 zero or negative price rows
- Unique stock codes: 5,305 (includes non-product codes). Countries: 43
- Note: the description column mixed numbers and text, so all text columns are converted to text before saving to parquet.

## Block 2: cleaning and exploration
- Status: done
- Result: 1,067,371 -> 1,001,935 rows (93.9% kept), 4,872 products. Rows with missing customer_id kept on purpose: 226,636.
- Rules (rows removed, in order): sales fully cancelled within 24 hours 1,410; cancelled invoices 19,494; quantity <= 0 3,457; price <= 0 2,750; non-product stock codes 4,715; exact duplicates 33,610.
- Decision: sale lines fully cancelled within 24 hours (same customer, product and quantity) are removed as order mistakes. This removed 214,753 units, including the two largest lines (invoice 581483 with 80,995 units and invoice 541431 with 74,215 units), which were each cancelled within about 20 minutes.
- Decision: other very large lines are kept. For example customer 13902 has five lines between 12,480 and 19,152 units, which look like real bulk orders. Their effect is checked later with a sensitivity test.
- Finding: the two fake top sellers (products 23843 and 23166) disappeared from the top 15, and the share of units from the top 15 products fell from 9.6% to 9.0%. Demand is spread over a long tail of products.
- Finding: demand is seasonal (peak from September to November) and drops to almost zero around the Christmas and New Year shutdown at the start of January in both years. There are only two yearly cycles in the data.
- Finding: the first and last week of the data are incomplete (data starts on a Tuesday and ends on a Friday) and are excluded in Block 3.
- Finding: the United Kingdom dominates sales by a wide margin.

## Block 3: weekly demand table and product segmentation
- Status: done
- Weekly table: 104 complete weeks (2009-12-07 to 2011-11-28), 4,838 products, 321,637 product-weeks. The first and last week are excluded because they are incomplete. 40.0% of product-weeks have zero sales (zeros are counted only between a product's first and last sale).
- ABC by revenue: A = 1,045 products (21.6% of products, 80% of revenue), B = 1,267 (26.2%, 15%), C = 2,526 (52.2%, 5%).
- Demand classes (Syntetos-Boylan): lumpy 2,214; erratic 1,571; intermittent 562; smooth 321; insufficient history 170. Only 6.6% of products have smooth demand.
- Clustering (KMeans, 5 features): best k = 2 with silhouette 0.311 (k = 3 to 6 gave 0.22 to 0.24). The two clusters are mostly "sparse low sellers" (1,485 products, median 1.5 units per week) and "regular sellers" (2,357 products, median 23 units per week). The cluster structure is weak and is not oversold.
- Modelling set: 2,042 forecastable products (at least 52 active weeks and 40 selling weeks) cover 81.1% of all units. Of these, 1,077 are erratic, 848 lumpy, 95 smooth and 22 intermittent.

## Block 4: baseline forecasts and time-based backtest
- Status: done
- Setup: one-week-ahead forecasts for the 2,042 forecastable products. Test period is the last 26 complete weeks (2011-06-06 to 2011-11-28). Every method is scored on the same 44,124 product-weeks (94.8% of actual units). Forecasts for a week only use earlier weeks.
- Metric: WAPE (total absolute error / total actual units) and bias (total error / total actual units).
- Overall WAPE: last week 0.814 (current practice), moving average of 4 weeks 0.696, moving average of 13 weeks 0.680 (best baseline), same week last year 1.099.
- Finding: averaging beats repeating last week by about 16%. The 13-week average under-forecasts by 13% in the test period because it lags the autumn ramp-up.
- Finding: the seasonal naive method over-forecasts by 28%. With only two years of data and a lower level in autumn 2011, copying last year does not work.
- Finding: errors are high for every demand class (best 0.43 for smooth products, about 0.85 to 0.96 for lumpy products), so point forecasts alone are not enough and prediction intervals are needed.
- Target for Block 5: beat WAPE 0.680.

## Block 5: gradient boosting quantile forecasts
- Status: done
- Model: histogram-based gradient boosting (scikit-learn HistGradientBoostingRegressor) with quantile loss, three models for p10, p50 and p90, trained on log(1 + units). Trained once on weeks before the test period (117,347 rows) and scored on 47,342 test rows. Features: lags (1, 2, 3, 4, 8, 13 weeks), rolling mean and standard deviation over 4 and 13 weeks, share of weeks with sales, week of year, month, and training-period product averages and price. No hyperparameter tuning was done on the test period.
- Result (same 44,124 product-weeks as the baselines): WAPE 0.631 for the median forecast, versus 0.680 for the best baseline (13-week average, about 7% better) and 0.814 for repeating last week (about 22% better). Better for ABC classes A, B and C and for erratic and lumpy products, and tied for smooth and intermittent products (only 95 and 22 products).
- Bias: the median forecast under-forecasts total units by about 30% (61% for lumpy products). This is expected for skewed demand, so ordering decisions use the upper quantile and not the median.
- Prediction interval p10 to p90: coverage 89.7% against a nominal 80%, so the intervals are conservative (wider than needed).
- Fairness check: the model predicts a median while the first baselines predict means, so a rolling 13-week median baseline was added for a like-for-like comparison.

## Block 6: ordering decisions and cost backtest
- Status: done (out-of-sample check follows in Block 6b)
- Setup: weekly order-up-to policy with carry-over stock, zero lead time, unmet demand is lost. Simulated on the same 44,124 product-weeks (2,042 products, 26 weeks). Five policies: repeat last week x k (current practice at k = 1), 13-week average x k, 13-week average + z x standard deviation, gradient boosting median x k, and gradient boosting median + z x (p90 - p50). k and z were swept over a grid and the best service-versus-capital curve was compared. Cost assumptions (not in the data): unit cost is 60% of price and holding costs 20% of stock value per year.
- Current practice (repeat last week, k = 1): fill rate 69.6%, 26.8% of product-weeks with a stockout, average capital tied up 122,369, total cost 529,998.
- Capital needed for a target fill rate: 90% fill needs 238,883 with the quantile model, 269,348 with the 13-week average plus safety stock (model about 11% lower) and 350,071 when repeating last week (about 32% lower). 95% fill needs 410,574, 467,386 (about 12% lower) and 673,467 (about 39% lower). Only the quantile model reached 98% fill within the grid (761,763).
- With the current capital (122,369): quantile model fill rate 79.2% and 15.2% of product-weeks with a stockout, against 76.3% and 21.2% for the best tuned baseline.
- Lowest total cost: quantile model 112,138 against 126,916 for the best tuned baseline (about 12% lower) and 158,574 for repeat last week x k. The 78.8% reduction against the untuned current practice (k = 1) is not a fair headline, because k = 1 is poorly tuned.
- Limitations: k and z were tuned on the same weeks used for evaluation (optimistic for every policy), the grid was too short for two policies (best value at the edge), lead time is zero, and the cost assumptions are not from the data.

## Block 6b: out-of-sample check of the ordering policies
- Status: done
- Setup: the 26 test weeks were split in half. The dial (k or z) of each policy was chosen on the first 13 weeks (2011-06-06 to 2011-08-29) and then fixed and evaluated on the last 13 weeks (2011-09-05 to 2011-11-28, the busy season; 22,140 product-weeks). The parameter grid was widened to 12.
- Current practice on the evaluation half (repeat last week, k = 1): fill rate 68.7%, capital 128,136, total cost 336,630.
- Dial chosen by lowest total cost: quantile model 66,915 (fill 97.8%), best tuned classical baseline (13-week average + z x standard deviation) 72,129 (fill 96.8%), 13-week average x k 79,603, repeat last week x k 102,165. The quantile model is 7.2% cheaper than the best baseline and 34.5% cheaper than repeating last week.
- Dial chosen as the smallest value reaching 95% fill on the tuning half: all good policies reached 95.5% to 95.6% on the evaluation half. Capital: quantile model 506,688, best baseline 520,627 (2.7% less), 13-week average x k 614,715 (17.6% less). Total cost: 74,486 against 78,011 (4.5% less).
- Conclusion: the advantage of the quantile model over a tuned classical safety-stock baseline is real but small out of sample (about 3% to 7%), smaller than the 11% to 12% seen in-sample. The advantage over simple rules is large.
- Caveats: cost assumptions (unit cost 60% of price, holding 20% per year) are not from the data, capital limits are ignored, the median x k policy hit the edge of the grid and should not be used.

## Block 7: dashboard, README and packaging
- Status: done
- Dashboard: Streamlit app with six tabs (overview, data and cleaning, forecast accuracy, product explorer, ordering simulation with a target fill-rate slider, method and limits). It only reads result files, so it adds no new calculations. Numbers in the dashboard match the terminal output.
- README: generated by src/make_readme.py from the result files, so every number in it comes from the files. Static parts (limitations, method, how to run) are written by hand.
- Packaging: requirements.txt created with pip freeze; data/ is excluded from version control.
- Not built yet: IoT stream (MQTT), API, Docker, LLM weekly report.
