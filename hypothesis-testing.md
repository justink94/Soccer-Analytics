## This hypothesis test will test the theory that shooting more often regardless of position will help yield more wins for a team. In other words, don't try and get into perfect position for every shot. Take as many as you can and you are bound to catch a defender or goal keeper off guard at some point.

```{r}
library(tidyverse)
```

### Create 'all teams' frame by binding together all 5 hypothesis test frames you made with the python hypo test function previously

```{r}
all_teams <- rbind(all_teams, spain_hypo)
```

### $H_0$ = Teams that shoot higher than average do not win more than average $\mu = 14.07$
### $H_1$ = Teams that shoot higher than average do win more than average $\mu > 14.07$
### $\alpha$ = 0.05

### We have a data frame of all teams that shot above average and their amounts of wins across Europe. The mean wins for all teams is 14.07. I am going to test to see whether teams that shot more have significantly more wins that teams that did not.

```{r}
shot_above = all_teams %>%
  filter(all_teams$`above average` == '1')
shot_below = all_teams %>%
  filter(all_teams$`above average` == '0')
```

### Even though our sample size is larger than 30, I am still going to use the Shapiro-Wilk normality test for the data. As we can see with an $\alpha$ = 0.05 and a p-value of 0.6649 we do not reject the null hypothesis of the data being normally distributed. So we can move forward with our test

```{r}
shapiro.test(shot_above$Wins)
```

### With an $\alpha$ of 0.05 and a p-value of 4.392e-05, we reject the null hypothesis of teams that shoot more than average do not win more than average. 

```{r}
t.test(shot_above$Wins, mu = 14.07, type = 'one tailed')
```

```{r}
ggplot(data = all_teams, aes(x = all_teams$`above average`, y = all_teams$Wins, fill = all_teams$`above average`))+
  geom_boxplot(lwd=0.7)+
  labs(title = 'Distributions of Wins for \nEuropean Club Teams', subtitle = 'By team shot totals', x = 'Total Shots Above Average', y = 'Number of Wins')+
  scale_x_discrete(labels = c('No', 'Yes'))+
  theme(plot.title= element_text(size=18,
                                   color="black",
                                   face="bold",
                                   family = "Tahoma"),
        plot.subtitle = element_text(size=10,
                                   color="black",
                                   face="bold",
                                   family = "Tahoma"),
        axis.text.x = element_text(family = 'Tahoma', face = 'bold'),
        axis.text.y = element_text(family = 'Tahoma', face = 'bold'),
        axis.title.x = element_text(face = 'bold', size = 12),
        axis.title.y = element_text(face = 'bold', size = 12),
        panel.background = element_rect(fill='grey'),
        axis.line = element_line(colour = "black"),
        panel.grid.major = element_line(colour = "white", size = (0.3)), legend.position="none",
        panel.border = element_rect(colour = "black", fill=NA, size=1))
```

![image](https://user-images.githubusercontent.com/70713627/211112661-74dd86b2-46d1-4014-a7b9-f4876b791418.png)

### We can see from this box plot, that we have a difference in wins between teams that shot below the average and shot above that average. The t test that we did proves that this difference is significant and proves that shooting more is a factor in winning more.
