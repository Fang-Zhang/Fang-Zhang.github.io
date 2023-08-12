import React, { Fragment, useState, useEffect } from 'react';
import Container from '@mui/material/Container';
import useMediaQuery from '@mui/material/useMediaQuery';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Carousel from 'react-slick';
import { useTranslation } from 'next-i18next';
import ImageThumbCard from '../Cards/ImageThumb';
import Title from '../Title';
import useStyle from './gallery-style';
import PortfolioData from './PortfolioData';

function Gallery() {
  const { classes, cx } = useStyle();
  const { t } = useTranslation('common');
  const [data, setData] = useState([]);
  const [isLoaded, setLoaded] = useState(true);
  const [filter, setFilter] = useState('all');
  const isMobile = useMediaQuery(theme => theme.breakpoints.down('sm'));

  useEffect(() => {
    setData(PortfolioData);
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
      const filteredData = PortfolioData.filter(item => item.category === name);
      setData(filteredData);
      setFilter(name);
    } else {
      setData(PortfolioData);
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
          {/* {t('profile-landing.gallery_title')} */}
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
            onClick={() => filterChildren('os')}
            className={filter === 'os' ? classes.selected : ''}
          >
            {t('profile-landing.gallery_cat_os')}
          </Button>
          <Button
            onClick={() => filterChildren('web')}
            className={filter === 'web' ? classes.selected : ''}
          >
            {t('profile-landing.gallery_cat_web')}
          </Button>
          <Button
            onClick={() => filterChildren('mobile')}
            className={filter === 'mobile' ? classes.selected : ''}
          >
            {t('profile-landing.gallery_cat_mobile')}
          </Button>
          <Button
            onClick={() => filterChildren('api')}
            className={filter === 'api' ? classes.selected : ''}
          >
            {t('profile-landing.gallery_cat_api')}
          </Button>
          <Button
            onClick={() => filterChildren('video')}
            className={filter === 'video' ? classes.selected : ''}
          >
            {t('profile-landing.gallery_cat_video')}
          </Button>
          <Button
            onClick={() => filterChildren('text')}
            className={filter === 'text' ? classes.selected : ''}
          >
            {t('profile-landing.gallery_cat_text')}
          </Button>
        </div>
        {!isMobile ? (
          <Fragment>
            <div className={classes.massonry}>
              {/* {data.map((item, index) => ( */}
              {data.slice(0, 6).map((item, index) => (
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
            {/* {data.map((item, index) => ( */}
            {data.slice(0, 6).map((item, index) => (
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
