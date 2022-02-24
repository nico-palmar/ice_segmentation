for i, (x, y) in enumerate(train_ds):
    # take a look at the range of pixel values given
    # x_mean = torch.mean(x, axis=0)
    # running_y = torch.cat((running_y, y.view(-1)), axis=0)
    if i == 500:
        break
    # max_y.append(y.max().item())
    y_vals += [*y.view(-1).tolist()]
    x_vals += [*x.view(-1).tolist()]
    # print(x.shape, y.shape)


    
    # fig, ax = plt.subplots(2)
    # ax[0].imshow(y.permute(1, 2, 0).detach())
    # ax[1].imshow(x.permute(1, 2, 0).detach())
    # plt.savefig(f'output/idx_{i}.png')
    # if i > 500:
    #     break

print(pd.Series(y_vals).value_counts())
# print(len(y_vals))
set_y = set(y_vals)
# print(len(set_y))
print(sorted(list(set_y)))
# print(set_y)

# set_x = set(x_vals)
# print(len(set_x))
# print(sorted(list(set_x)))

# ax = sns.histplot(y_vals)
# ax.set_title('X values histplot adjusted')

# plt.savefig(f'output/im_first_one_hundred_adjusted_vals.png')

# sns.histplot(max_y)
# plt.savefig(f'output/max_mask_y.png')

# idx_21_series = pd.Series(train_ds[21][0].view(-1))
# descr = idx_21_series.describe()
# print(descr)


# print(len(img_paths), len(mask_paths))
# print(len(train_x_path), len(valid_x_path), len(test_x_path))