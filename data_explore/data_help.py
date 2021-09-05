from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.metrics import roc_auc_score as AUC
import matplotlib.pyplot as plt
import numpy as np

# Functions to help with data exploration

def player_stat_grouping(df, foi):
    """
    Return a dataframe with the means and stds for a player's fields of interest (foi) during a year
    
    df: Dataframe of data. Assumed to have column with title 'year', 'unique', and foi column names
    foi: Fields of interests to group on. Could be either string or list
    """
    if type(foi) != list:
        foi = [foi]
    group = df[['year', 'unique_id'] + foi].groupby(['year', 'unique_id'])
    return group.mean(), group.std()

def plot_multiple_yearly_hists(df, position, fois, histby='year', **kwargs):
    """
    Plot multiple histograms for each year across multiple particular field of interest (foi) for a given position.
    
    df: Dataframe of data. Assumed to have column with title 'year', 'pos', and foi column
    fois: field of interest. Must be a column within df
    kwargs: See here for valid kwargs https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html
    """
    for foi in fois:
        plot_yearly_hist(df, position, foi, histby=histby, **kwargs)

def plot_yearly_hist(df, position, foi, histby='year', **kwargs):
    """
    Plot a histogram for each year for a particular field of interest (foi) for a given position.
    
    df: Dataframe of data. Assumed to have column with title 'year', 'pos', and foi column
    foi: field of interest. Must be a column within df
    kwargs: See here for valid kwargs https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html
    """
    year_foi = df[(df['pos'] == position)][[histby, foi]]
    axs = year_foi.hist(foi, by=histby, sharex=True, figsize=(10, 10), **kwargs)
    # Get means for plotting
    means = year_foi.groupby(histby).mean().reset_index(drop=True)
    # Get stds for plotting
    stds = year_foi.groupby(histby).std().reset_index(drop=True)
    for i, axes in enumerate(axs):
        for j, ax in enumerate(axes):
            curr_idx = i * axs.shape[0] + j
            # Have to try to draw lines on all plots because they sometimes get a weird structure
            curr_idx = means.shape[0] - 1 if curr_idx >= means.shape[0] else curr_idx
            mean = means.iloc[curr_idx][foi]
            std = stds.iloc[curr_idx][foi]
            ax.axvline(mean, color='r', linestyle='dashed', label = 'mean')
            ax.axvline(mean + std, color='y', linestyle='dashed', label='1 std dev')
            ax.axvline(mean - std, color='y', linestyle='dashed')
            
            # If last gather label for legend
            if i == axs.shape[0] - 1 and j == axes.shape[0] - 1:
                handles, labels = ax.get_legend_handles_labels()
                
    plt.figlegend(handles, labels, loc='upper right')
    plt.suptitle(position + ' ' + foi + ' Histograms')
    plt.show()

def compare_splits(m, df, split_fns, input_vars=None, **kwargs):
    """
    Use split_fns to form splits on df which can then be compared using a model m to see
    if a covariate shift exists between the various splits. returns 2d array summarizing AUC
    results of the splits.
    
    m: model to use
    df: dataframe to split over
    split_fns: array of functions used to split over the dataframe. should take in input variable
               df which is the passed dataframe and returns a boolean series over it
    input_vars: Variables to actually pass to the model for checking covariate shift.
    kwargs: see check_covariate_shift for possible args
    
    Returns: nxn array of results where entry (i, j) is the AUC score for i's distinction from j
    """
    # Copy df
    df = df.copy()
    
    # Constant for target column name
    target_column = 'target'
    
    # Store results
    results = np.zeros((len(split_fns), len(split_fns)))
    for i, split_fn1 in enumerate(split_fns):
        for j, split_fn2 in enumerate(split_fns):
            if i == j:
                results[i][j] = 1
                continue
            # Assume results are symmetric
            elif i > j:
                continue
                
            class_1 = np.where(split_fn1(df))[0]
            class_2 = np.where(split_fn2(df))[0]
            
            df.loc[class_1, target_column] = 0
            df.loc[class_2, target_column] = 1
            
            all_rows = np.append(class_1, class_2)
            
            if input_vars is not None:
                x = df.loc[all_rows, input_vars].values
            else:
                x = df.loc[all_rows, :].values
            
            y = df.loc[all_rows, target_column].values
            
            shift = check_covariate_shift(m, x, y, **kwargs)
            results[i][j] = shift
            # Assume symmetry
            results[j][i] = shift
            
    return results

def check_covariate_shift(m, x, y, num_splits=5, shuffle=True, **kwargs):
    """
    checks the covariate shifts of the
    """
    predictions = np.zeros(y.shape)

    skf = SKF(n_splits=num_splits, **kwargs)
    for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
        X_train, X_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        m.fit(X_train, y_train)
        probs = m.predict_proba(X_test)[:, 1] #calculating the probability
        predictions[test_idx] = probs
        
    return AUC(y, predictions)

def heatmap(values, xlabels, ylabels, title, figsize=(10, 10), decimals=2):
    """
    Generate a heatmap for values using provided xlabels, ylabels, and title
    
    values: the values to generate the heatmap for
    xlabels: the x labels to use on the heatmap
    ylabels: the y labels to use on the heatmap
    title: the title for the heatmap
    figsize: the size of the figure for the heatmap
    decimals: the number of decimals to round the text to
    """
    # code from here: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(values)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(xlabels)):
        for j in range(len(ylabels)):
            text = ax.text(j, i, round(values[i, j], decimals),
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()

def auc_distance_scatter(splits, title):
    # Plot how distance changes 
    results = [[] for dist in range(splits.shape[0])]
    for i in range(splits.shape[0]):
        for j in range(splits.shape[1]):
            if i >= j:
                continue
            distance = abs(i - j)
            results[distance].append(splits[i][j])
    
    means = [np.mean(result) for result in results]
    
    xs = []
    ys = []
    for i, dist in enumerate(results):
        for y in dist:
            ys.append(y)
            xs.append(i)
            
    plt.scatter(xs, ys)
    plt.plot(range(len(means)), means, color='r')
    plt.xlabel('Distance between Years')
    plt.ylabel('AUC Score')
    plt.ylim(bottom=0, top=1)
    plt.xticks([x for x in range(splits.shape[0])])
    plt.title(title)
    plt.show()