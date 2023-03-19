import React, { Fragment, useState, useEffect } from 'react';
import Container from '@mui/material/Container';
import useMediaQuery from '@mui/material/useMediaQuery';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Carousel from 'react-slick';
import { useTranslation } from 'next-i18next';
import imgAPI from '/public/images/imgAPI';
import ImageThumbCard from '../Cards/ImageThumb';
import Title from '../Title';
import useStyle from './gallery-style';

const portfolio = [
  {
    img: imgAPI.profile[4],
    title: 'work01',
    link: 'https://fang-zhang.com',
    size: 'short',
    category: 'cat1'
  },
  {
    img: imgAPI.profile[3],
    title: 'work02',
    link: 'https://fang-zhang.com',
    size: 'short',
    category: 'cat2'
  },
  {
    img: imgAPI.profile[5],
    title: 'work03',
    link: 'https://fang-zhang.com',
    size: 'long',
    category: 'cat3'
  },
  {
    img: imgAPI.profile[6],
    title: 'work04',
    link: 'https://fang-zhang.com',
    size: 'short',
    category: 'cat4'
  },
  {
    img: imgAPI.profile[8],
    title: 'work05',
    link: 'https://fang-zhang.com',
    size: 'short',
    category: 'cat5'
  }
];

function Gallery() {
  const { classes, cx } = useStyle();
  const { t } = useTranslation('common');
  const [data, setData] = useState([]);
  const [isLoaded, setLoaded] = useState(true);
  const [filter, setFilter] = useState('all');
  const isMobile = useMediaQuery(theme => theme.breakpoints.down('sm'));

  useEffect(() => {
    setData(portfolio);
  }, []);

  const settings = {
    dots: false,
    infinite: false,
    speed: 500,
    autoplay: false,
    slidesToShow: 1,
    arrows: false
  };

  const filterChildren = name => {
    if (name !== 'all') {
      const filteredData = portfolio.filter(item => item.category === name);
      setData(filteredData);
      setFilter(name);
    } else {
      setData(portfolio);
      setFilter('all');
    }

    setLoaded(false);
    setTimeout(() => {
      setLoaded(true);
    }, 700);
  };

  return (
    <div className={classes.root}>
      <Container>
        <Title>
          {t('profile-landing.gallery_title')}
          <strong>
            {t('profile-landing.gallery_titleBold')}
          </strong>
        </Title>
        <div className={classes.filter}>
          <Button
            onClick={() => filterChildren('all')}
            className={filter === 'all' ? classes.selected : ''}
          >
            {t('profile-landing.gallery_cat_all')}
          </Button>
          <Button
            onClick={() => filterChildren('cat1')}
            className={filter === 'cat1' ? classes.selected : ''}
          >
            {t('profile-landing.gallery_cat_cat1')}
          </Button>
          <Button
            onClick={() => filterChildren('cat2')}
            className={filter === 'cat2' ? classes.selected : ''}
          >
            {t('profile-landing.gallery_cat_cat2')}
          </Button>
          <Button
            onClick={() => filterChildren('cat3')}
            className={filter === 'cat3' ? classes.selected : ''}
          >
            {t('profile-landing.gallery_cat_cat3')}
          </Button>
          <Button
            onClick={() => filterChildren('cat4')}
            className={filter === 'cat4' ? classes.selected : ''}
          >
            {t('profile-landing.gallery_cat_cat4')}
          </Button>
          <Button
            onClick={() => filterChildren('cat5')}
            className={filter === 'cat5' ? classes.selected : ''}
          >
            {t('profile-landing.gallery_cat_cat5')}
          </Button>
          <Button
            onClick={() => filterChildren('cat6')}
            className={filter === 'cat6' ? classes.selected : ''}
          >
            {t('profile-landing.gallery_cat_cat6')}
          </Button>
        </div>
        {!isMobile ? (
          <Fragment>
            <div className={classes.massonry}>
              {data.map((item, index) => (
                <div
                  className={cx(classes.item, isLoaded && classes.loaded)}
                  key={index.toString()}
                  style={{ transitionDuration: index / 4 + 's' }}
                >
                  <ImageThumbCard
                    img={item.img}
                    title={item.title}
                    link={item.link}
                    size={item.size}
                  />
                </div>
              ))}
            </div>
            {data.length < 1 && <Typography variant="button" component="p" align="center">{t('profile-landing.gallery_nodata')}</Typography>}
          </Fragment>
        ) : (
          <Carousel {...settings}>
            {data.map((item, index) => (
              <div
                className={classes.itemCarousel}
                key={index.toString()}
              >
                <ImageThumbCard
                  img={item.img}
                  title={item.title}
                  link={item.link}
                  size={item.size}
                />
              </div>
            ))}
          </Carousel>
        )}
      </Container>
    </div>
  );
}

export default Gallery;
